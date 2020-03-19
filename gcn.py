import numpy as np
import pretty_errors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import fastNLP
from fastNLP import (SGD, AccuracyMetric, Adam, CrossEntropyLoss, DataSet,
                     DataSetIter, SpanFPreRecMetric, Trainer, Vocabulary)
from fastNLP.core.callback import FitlogCallback
from fastNLP.core.metrics import MetricBase
from fastNLP.embeddings import StaticEmbedding
from fastNLP.modules import LSTM,MLP
from tree import *
class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix):
        super().__init__()

        in_dim = args.hidden_dim
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = MLP([in_dim, args.num_class])

    def forward(self, mask, chars, aspect, seq_len, head, posidx, postidx, depidx):
        output = self.gcn_model(
            mask, chars, aspect, seq_len, head, posidx, postidx, depidx
        )["pred"]
        logit = self.classifier(output)
        return {"pred": logit}


class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args

        self.emb_matrix = emb_matrix
        self.tok_emb = nn.Embedding(args.tok_size, args.tok_dim, padding_idx=0)
        if emb_matrix is not None:
            self.tok_emb = emb_matrix  
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0)
            if args.pos_dim > 0
            else None
        )
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )
        embedding = (self.tok_emb, self.pos_emb, self.post_emb)

        # gcn
        self.gcn = GCN(args, embedding, args.hidden_dim)

    def forward(self, mask, chars, aspect, seq_len, head, posidx, postidx, depidx):
        tok = chars
        head = head
        mask = mask.to(torch.float32)
        seq_len = seq_len
        maxlen = max(seq_len)

        def inputs_to_tree_reps(head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i])
                     for i in range(len(l))]
            adj = [
                tree_to_adj(
                    maxlen, tree, directed=self.args.direct, self_loop=self.args.loop
                ).reshape(1, maxlen, maxlen)
                for tree in trees
            ]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda()

        adj = inputs_to_tree_reps(head, tok, seq_len)
        h = self.gcn(
            adj, mask, chars, aspect, seq_len, head, posidx, postidx, depidx
        )

        asp_num = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)
        output = (h["pred"] * mask).sum(dim=1) / asp_num

        return {"pred": output}


class GCN(nn.Module):
    def __init__(self, args, embedding, hid_dim):
        super(GCN, self).__init__()

        self.args = args
        self.layers = args.num_layers
        self.mem_dim = hid_dim
        self.in_dim = args.tok_dim + args.pos_dim + args.post_dim
        self.tok_emb, self.pos_emb, self.post_emb = embedding
        # drop out
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # lstm
        input_size = self.in_dim
        self.rnn = LSTM(
            input_size,
            args.rnn_hidden,
            args.rnn_layers,
            batch_first=True,
            dropout=args.rnn_dropout,
            bidirectional=args.bidirect,
        )
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden

        # gcn layer
        self.G = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = [self.in_dim, self.mem_dim][layer != 0]
            self.G.append(MLP([input_dim, self.mem_dim]))


    def forward(self, adj, mask, chars, aspect, seq_len, head, posidx, postidx, depidx):
        tok = chars
        pos = posidx
        post = postidx
        seq_len = seq_len

        word_embs = self.tok_emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn
        total_layers = self.args.rnn_layers * 2 if self.args.bidirect else self.args.rnn_layers
        h0 = c0 = Variable(
            torch.zeros([total_layers, tok.size()[0], self.args.rnn_hidden]),
            requires_grad=False,
        )
        h0 = h0.cuda()
        c0 = c0.cuda()

        rnn_output, _ = self.rnn(embs, h0=h0, c0=c0)
        gcn_input = self.rnn_drop(rnn_output)

        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  
        for l in range(self.layers):
            AX = adj.bmm(gcn_input)
            AXW = self.G[l](AX)
            AXW = F.relu(AXW / denom)
            gcn_input = self.gcn_drop(AXW) if l < self.layers - 1 else AXW
        return {"pred": gcn_input}

