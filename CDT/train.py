'''
Author: Roger
Date: 2020-11-30 12:24:37
LastEditors  : Roger
LastEditTime : 2021-02-05 10:44:41
Description: CDT train
'''
import argparse

import fitlog
from fastNLP import (AccuracyMetric, Adam, Callback, CrossEntropyLoss, Trainer,
                     Vocabulary)
from fastNLP.core.callback import FitlogCallback
from fastNLP.core.metrics import MetricBase
from fastNLP.embeddings import StaticEmbedding
from sklearn import metrics as m
from torch.utils.tensorboard import SummaryWriter

from cdt_model import GCNClassifier
from data_loader import CDTPipe
from tree import *

# fitlog.debug()
fitlog.commit(__file__)  # auto commit your codes
fitlog.set_log_dir("./logs")

parse = argparse.ArgumentParser()
parse.add_argument("--data_dir", type=str)
parse.add_argument("--tok_dim",
                   type=int,
                   default=300,
                   help="token embedding dimension.")
parse.add_argument("--post_dim",
                   type=int,
                   default=30,
                   help="Position embedding dimension.")
parse.add_argument("--pos_dim",
                   type=int,
                   default=30,
                   help="Pos embedding dimension.")
parse.add_argument("--hidden_dim", type=int, default=50, help="GCN mem dim.")
parse.add_argument("--num_layers",
                   type=int,
                   default=2,
                   help="Num of GCN layers.")
parse.add_argument("--num_class",
                   type=int,
                   default=3,
                   help="Num of sentiment class.")

parse.add_argument("--input_dropout",
                   type=float,
                   default=0.5,
                   help="Input dropout rate.")
parse.add_argument("--gcn_dropout",
                   type=float,
                   default=0.1,
                   help="GCN layer dropout rate.")
parse.add_argument("--direct", default=False)
parse.add_argument("--loop", default=True)

parse.add_argument("--bidirect", default=True, help="Do use bi-RNN layer.")
parse.add_argument("--rnn_hidden",
                   type=int,
                   default=50,
                   help="RNN hidden state size.")
parse.add_argument("--rnn_layers",
                   type=int,
                   default=1,
                   help="Number of RNN layers.")
parse.add_argument("--rnn_dropout",
                   type=float,
                   default=0.1,
                   help="RNN dropout rate.")

parse.add_argument("--lr", type=float, default=0.01, help="learning rate.")
parse.add_argument("--num_epoch",
                   type=int,
                   default=100,
                   help="Number of total training epochs.")
parse.add_argument("--batch_size",
                   type=int,
                   default=32,
                   help="Training batch size.")

parse.add_argument(
    "--save_dir",
    type=str,
    default="./save_models",
    help="Root dir for saving models.",
)
args = parse.parse_args()
fitlog.add_hyper(args)


data_bundle = CDTPipe().process_from_file(args.data_dir)
max_len = max(max(data_bundle.get_dataset("train")["seq_len"]),
              max(data_bundle.get_dataset("test")["seq_len"]))
post_vocab = Vocabulary()
post_vocab.add_word_lst(list(range(-max_len, max_len)))
data_bundle.set_vocab(post_vocab, "post")
for name, dataset in data_bundle.datasets.items():
    data_bundle.get_vocab("post").index_dataset(dataset, field_name="post")

args.tok_size = len(data_bundle.get_vocab("words"))
args.pos_size = len(data_bundle.get_vocab("pos"))
args.post_size = len(data_bundle.get_vocab("post"))

embed = StaticEmbedding(data_bundle.get_vocab(
    "words"), model_dir_or_name="en", requires_grad=False)

model = GCNClassifier(args, embed)

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
        return {"F1": f1_score}


class tensorboardcallback(Callback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def on_backward_begin(self, loss):
        self.writer.add_scalar('training loss', loss, self.step)

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        self.writer.add_scalar('test acc',
                               eval_result["AccuracyMetric"]["acc"],
                               self.step)
        self.writer.add_scalar('test f1', eval_result["f1metric"]["F1"],
                               self.step)


writer = SummaryWriter()
tensorboard_callback = tensorboardcallback(writer)

optimizer = Adam(lr=args.lr, weight_decay=0)
acc = AccuracyMetric()
f1 = f1metric()
loss = CrossEntropyLoss()
trainer = Trainer(
    data_bundle.get_dataset("train"),
    model,
    optimizer=optimizer,
    loss=loss,
    batch_size=args.batch_size,
    n_epochs=args.num_epoch,
    dev_data=data_bundle.get_dataset("test"),
    metrics=[acc, f1],
    save_path=args.save_dir,
    device=0,
    callbacks=[FitlogCallback(log_loss_every=20), tensorboard_callback],
)
trainer.train()
fitlog.finish()  # finish the logging
