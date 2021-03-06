import sys
import re
from time import time
from os.path import isfile
from parameters import *
from collections import defaultdict

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
zeros = lambda *x: torch.zeros(*x).cuda() if CUDA else torch.zeros

def normalize(x):
    # x = re.sub("[^ ,.?!a-zA-Z0-9\u3131-\u318E\uAC00-\uD7A3]+", " ", x)
    x = re.sub("(?=[,.?!])", " ", x)
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, norm = True):
    if norm:
        x = normalize(x)
    if UNIT == "char":
        return re.sub(" ", "", x)
    if UNIT in ("word", "sent"):
        return x.split(" ")

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tti = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tti[line] = len(tti)
    fo.close()
    return tti

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    itt = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        itt.append(line)
    fo.close()
    return itt

def load_checkpoint(filename, model = None):
    print("loading %s" % filename)
    checkpoint = torch.load(filename)
    if model:
        model.enc.load_state_dict(checkpoint["enc_state_dict"])
        model.dec.load_state_dict(checkpoint["dec_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        checkpoint = {}
        checkpoint["enc_state_dict"] = model.enc.state_dict()
        checkpoint["dec_state_dict"] = model.dec.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved %s" % filename)

def save_loss(filename, epoch, loss_array):
    fo = open(filename + ".epoch%d.loss" % epoch, "w")
    fo.write("\n".join(map(str, loss_array)) + "\n")
    fo.close()

def maskset(x):
    if type(x) == torch.Tensor:
        mask = x.eq(PAD_IDX)
        lens = x.size(1) - mask.sum(1)
        return mask, lens
    mask = Tensor([[1] * i + [PAD_IDX] * (x[0] - i) for i in x]).eq(PAD_IDX)
    return mask, x

def mat2csv(m, ch = True, rh = False, delim ="\t"):
    v = "%%.%df" % NUM_DIGITS
    if ch: # column header
        csv = delim.join(map(str, m[0])) + "\n" # source sequence
    for row in m[ch:]:
        if rh: # row header
            csv += str(row[0]) + delim # target sequence
        csv += delim.join([v % x for x in row[rh:]]) + "\n"
    return csv

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0