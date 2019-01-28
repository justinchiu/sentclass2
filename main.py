# python main.py --flat-data --devid 0 --data sst --fine-grain --ebsz 512 --rnn-sz 256 --lr 0.01 --dp 0.2 --bsz 128 --model lstmmax 
# python main.py --flat-data --devid 0 --data sst --fine-grain --ebsz 512 --rnn-sz 256 --lr 0.003 --dp 0.2 --bsz 128 --model crflstmlstm
 
import argparse

import torch
import torch.optim as optim

import torchtext
from torchtext.data import BucketIterator, Field
from torchtext.vocab import GloVe

from sentclass.models.lstm import Lstm
from sentclass.models.crflstmdiag import CrfLstmDiag
from sentclass.models.crfemblstm import CrfEmbLstm
from sentclass.models.crflstmlstm import CrfLstmLstm
from sentclass.models.crfneg import CrfNeg

import json

torch.backends.cudnn.enabled = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        #default="data",
        default="~/research/GCAE/acsa-restaurant-large",
        type=str,
    )

    parser.add_argument("--devid", default=-1, type=int)

    parser.add_argument("--flat-data", action="store_true", default=False)
    parser.add_argument("--data", choices=["semeval", "sst"]) 
    parser.add_argument("--fine-grained", action="store_true") 
    parser.add_argument("--train-subtrees", action="store_true") 

    parser.add_argument("--bsz", default=33, type=int)
    parser.add_argument("--ebsz", default=150, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--once", action="store_true")

    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=1, type=float)
    parser.add_argument("--pat", default=0, type=int)
    parser.add_argument("--dp", default=0.2, type=float)
    parser.add_argument("--wdp", default=0, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)

    parser.add_argument("--optim", choices=["Adam", "SGD"])

    # Adam
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD
    parser.add_argument("--mom", type=float, default=0)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    # Model
    parser.add_argument(
        "--model",
        choices=[
            "lstmmax", "lstmfinal",
            "crflstmdiag", "crfemblstm", "crflstmlstm",
            "crfneg",
        ],
        default="lstmfinal"
    )

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=300, type=int)
    parser.add_argument("--rnn-sz", default=50, type=int)

    parser.add_argument("--save", action="store_true")

    parser.add_argument("--re", default=100, type=int)

    parser.add_argument("--seed", default=1111, type=int)
    return parser.parse_args()


args = get_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid >= 0 else "cpu")

# Data
TEXT, ASPECT, SENTIMENT, train, valid, test = None, None, None, None, None, None
if args.data == "semeval":
    import sentclass.semeval as data

    TEXT, ASPECT, SENTIMENT = data.make_fields()
    train, valid, test = data.SemevalDataset.splits(
        TEXT, ASPECT, SENTIMENT, flat=args.flat_data, path=args.filepath,
        train="acsa_train.json.train", validation="acsa_train.json.valid", test="acsa_test.json",
    )
    data.build_vocab(TEXT, ASPECT, SENTIMENT, train, valid, test)
elif args.data == "sst":
    TEXT, SENTIMENT = (
        Field(tokenize="spacy", lower=True, include_lengths=True, batch_first=True, init_token="<bos>", eos_token="<eos>"),
        Field(lower=True, is_target=True, unk_token=None, pad_token=None, batch_first=True,),
    )
    train, valid, test = torchtext.datasets.SST.splits(
        TEXT, SENTIMENT, 
        fine_grained = args.fine_grained,
        train_subtrees = args.train_subtrees,
        #filter_pred=lambda ex: ex.label[0] != 'neutral',
    )
    TEXT.build_vocab(train, valid, test)
    SENTIMENT.build_vocab(train)
TEXT.vocab.load_vectors(vectors=GloVe(name="840B"))

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test),
    batch_sizes = (args.bsz, args.ebsz, args.ebsz),
    device = device,
    repeat = False,
    sort_within_batch = True,
)

# Model
if args.model == "lstmfinal":
    assert(args.flat_data)
    model = Lstm(
        V       = TEXT.vocab,
        A       = ASPECT and ASPECT.vocab,
        S       = SENTIMENT.vocab,
        final   = True,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "lstmmax":
    assert(args.flat_data)
    model = Lstm(
        V       = TEXT.vocab,
        A       = ASPECT and ASPECT.vocab,
        S       = SENTIMENT.vocab,
        final   = False,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crflstmdiag":
    assert(args.flat_data)
    model = CrfLstmDiag(
        V       = TEXT.vocab,
        A       = ASPECT and ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crfemblstm":
    assert(args.flat_data)
    model = CrfEmbLstm(
        V       = TEXT.vocab,
        A       = ASPECT and ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crflstmlstm":
    assert(args.flat_data)
    model = CrfLstmLstm(
        V       = TEXT.vocab,
        A       = ASPECT and ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crfneg":
    assert(args.flat_data)
    model = CrfNeg(
        V       = TEXT.vocab,
        A       = ASPECT and ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )

model.to(device)
print(model)

params = list(model.parameters())

optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))

best_val = 0
for e in range(args.epochs):
    print(f"Epoch {e} lr {optimizer.param_groups[0]['lr']}")
    train_iter.init_epoch()
    # Train
    train_loss, tntok = model.train_epoch(
        diter     = train_iter,
        clip      = args.clip,
        re        = args.re,
        optimizer = optimizer,
        once      = args.once,
    )

    # Validate
    valid_loss, ntok = model.validate(valid_iter)

    # Accuracy on train
    train_acc = model.acc(train_iter)
    # Accuracy on Valid
    valid_acc = model.acc(valid_iter, skip0=False)
    #valid_f1 = model.f1(asp_valid_iter)
    test_acc = model.acc(test_iter, skip0=False)
    #test_f1 = model.f1(asp_test_iter)

    # Report
    print(f"Epoch {e}")
    print(f"train loss: {train_loss / tntok} train acc: {train_acc}")
    print(f"valid loss: {valid_loss / ntok} valid acc: {valid_acc}")# valid f1: {valid_f1}")
    print(f"test acc: {test_acc}")# test f1: {test_f1}")

    if args.save and valid_acc > best_val:
        best_val = valid_acc
        savestring = f"saves/{args.model}/{args.model}-lr{args.lr}-nl{args.nlayers}-rnnsz{args.rnn_sz}-dp{args.dp}-va{valid_acc}-ta{test_acc}.pt"
        torch.save(model, savestring)
