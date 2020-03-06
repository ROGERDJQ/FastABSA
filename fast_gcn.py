import argparse
import json
import os
import random

import fitlog
import numpy as np
import pretty_errors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as m
from torch.autograd import Variable

import fastNLP
from fastNLP import (SGD, AccuracyMetric, Adam, CrossEntropyLoss, DataSet,
                     DataSetIter, SpanFPreRecMetric, Trainer, Vocabulary)
from fastNLP.core.callback import FitlogCallback
from fastNLP.core.metrics import MetricBase
from fastNLP.embeddings import StaticEmbedding
from fastNLP.modules import LSTM,MLP

from tree import *
from gcn import GCNClassifier

fitlog.commit(__file__)  # auto commit your codes
fitlog.add_hyper_in_file(__file__)  # record your hyperparameters



parse = argparse.ArgumentParser()
parse.add_argument(
    "--data_dir",
    type=str,
    default="dataset/Restaurants",
)
parse.add_argument(
    "--glove_dir",
    type=str,
    default="dataset/glove/glove.840B.300d.txt",
)
parse.add_argument("--tok_dim", type=int, default=300,
                   help="token embedding dimension.")
parse.add_argument(
    "--post_dim", type=int, default=30, help="Position embedding dimension."
)
parse.add_argument("--pos_dim", type=int, default=30,
                   help="Pos embedding dimension.")
parse.add_argument("--hidden_dim", type=int, default=50, help="GCN mem dim.")
parse.add_argument("--num_layers", type=int, default=2,
                   help="Num of GCN layers.")
parse.add_argument("--num_class", type=int, default=3,
                   help="Num of sentiment class.")

parse.add_argument(
    "--input_dropout", type=float, default=0.5, help="Input dropout rate."
)
parse.add_argument(
    "--gcn_dropout", type=float, default=0.1, help="GCN layer dropout rate."
)
parse.add_argument("--direct", default=False)
parse.add_argument("--loop", default=True)

parse.add_argument("--bidirect", default=True, help="Do use bi-RNN layer.")
parse.add_argument("--rnn_hidden", type=int, default=50,
                   help="RNN hidden state size.")
parse.add_argument("--rnn_layers", type=int, default=1,
                   help="Number of RNN layers.")
parse.add_argument("--rnn_dropout", type=float,
                   default=0.1, help="RNN dropout rate.")

parse.add_argument("--lr", type=float, default=0.01, help="learning rate.")
parse.add_argument(
    "--num_epoch", type=int, default=100, help="Number of total training epochs."
)
parse.add_argument("--batch_size", type=int, default=32,
                   help="Training batch size.")

parse.add_argument(
    "--save_dir",
    type=str,
    default="./fast_saved_models",
    help="Root dir for saving models.",
)
parse.add_argument("--seed", type=int, default=0)
args = parse.parse_args()



torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def read_json(path, is_low=True):
    # 将json 读取为 dataset, 增加seq_len,post,mask field
    with open(path, "r", encoding="utf8") as f:
        raw_data = json.load(f)
    data = []
    for d in raw_data:
        tok = d["token"] if not is_low else [w.lower() for w in d["token"]]
        pos = d["pos"]
        head = d["head"]
        deprel = d["deprel"]
        for aspect in d["aspects"]:
            asp = [aspect["term"],["urgrn"]][len(aspect["term"])==0]  # for Restaurants16
            fidx = aspect["from"]
            tidx = aspect["to"]
            pol = aspect["polarity"]
            data.append([tok, pos, head, deprel, asp, fidx, tidx, pol])

    fields = ["tok", "pos", "head", "deprel", "asp", "fidx", "tidx", "pol"]
    dataset = DataSet(dict(zip(fields, zip(*data))))

    def get_post(instance):
        fidx = instance["fidx"]
        tidx = instance["tidx"]
        seq_len = instance["seq_len"]
        return (
            [i - fidx for i in range(fidx)]
            + [0 for _ in range(fidx, tidx)]
            + [i - tidx + 1 for i in range(tidx, seq_len)]
        )

    def get_mask(instance):
        if not instance["fidx"]==instance["tidx"]:
            return [[0, 1][i in range(instance["fidx"], instance["tidx"])] for i in range(instance["seq_len"])]
        else:
            return [1 for _ in range(instance["seq_len"])]

    dataset.apply(lambda line: len(line["tok"]), new_field_name="seq_len")
    dataset.apply(get_post, new_field_name="post")
    dataset.apply(get_mask, new_field_name="mask")
    return dataset


def get_vocab(trainset, testset):
    # 构建vocab以及word2idx
    #tok
    tok_vocab = Vocabulary()
    tok_vocab.from_dataset(trainset, field_name="tok",
                           no_create_entry_dataset=testset)
    tok_vocab.index_dataset(
        trainset, testset, field_name="tok", new_field_name="chars")
    tok_vocab.index_dataset(
        trainset, testset, field_name="asp", new_field_name="aspect"
    )
    # deprel
    dep_vocab = Vocabulary()
    dep_vocab.from_dataset(trainset, field_name="deprel")
    dep_vocab.index_dataset(
        trainset, testset, field_name="deprel", new_field_name="depidx"
    )
    # pol(target)
    pol_vocab = Vocabulary(padding=None, unknown=None)
    pol_vocab.from_dataset(trainset, field_name="pol")
    pol_vocab.index_dataset(
        trainset, testset, field_name="pol", new_field_name="target"
    )
    # pos
    pos_vocab = Vocabulary()
    pos_vocab.from_dataset(trainset, field_name="pos")
    pos_vocab.index_dataset(
        trainset, testset, field_name="pos", new_field_name="posidx"
    )
    # post
    max_len = max(max(trainset["seq_len"]), max(testset["seq_len"]))
    post_vocab = Vocabulary()
    post_vocab.add_word_lst(list(range(-max_len, max_len)))
    post_vocab.index_dataset(
        trainset, testset, field_name="post", new_field_name="postidx"
    )
    return tok_vocab, pos_vocab, post_vocab, trainset, testset


trainset = read_json(args.data_dir + "/train.json")
testset = read_json(args.data_dir + "/test.json")
tok_vocab, pos_vocab, post_vocab, trainset, testset = get_vocab(
    trainset, testset)
trainset.set_input(
    "head", "seq_len", "mask", "chars", "aspect", "depidx", "posidx", "postidx"
)
testset.set_input(
    "head", "seq_len", "mask", "chars", "aspect", "depidx", "posidx", "postidx"
)
trainset.set_target("target")
testset.set_target("target")

args.tok_size = len(tok_vocab)
args.pos_size = len(pos_vocab)
args.post_size = len(post_vocab)


embed = StaticEmbedding(
    tok_vocab, model_dir_or_name=args.glove_dir, requires_grad=False
)

cla = GCNClassifier(args, embed)
cla.cuda()


class f1metric(MetricBase):
    def __init__(self, label=None, pred=None):
        super().__init__()
        self._init_param_map(target=label, pred=pred)
        self.l = []
        self.p = []

    def evaluate(self, target, pred):
        self.l.extend(target.tolist())
        pred = pred.argmax(dim=-1).tolist()
        self.p.extend(pred)

    def get_metric(self, reset=True):
        f1_score = m.f1_score(self.l, self.p, average="macro")
        if reset:
            self.l = []
            self.p = []
        return {"f1:": f1_score}


optimizer = Adam(lr=args.lr, weight_decay=0)
acc = AccuracyMetric()
f1 = f1metric()
loss = CrossEntropyLoss()
trainer = Trainer(
    trainset,
    cla,
    optimizer=optimizer,
    loss=loss,
    batch_size=args.batch_size,
    n_epochs=args.num_epoch,
    dev_data=testset,
    metrics=[acc, f1],
    save_path=args.save_dir,
    callbacks=[FitlogCallback(log_loss_every=5)],
)
trainer.train()

fitlog.finish()  # finish the logging
