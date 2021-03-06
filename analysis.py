import torch

from torchtext.vocab import GloVe
from torchtext.data import Batch

from sentclass.models.lstmfinal import LstmFinal
from sentclass.models.crflstmdiag import CrfLstmDiag
from sentclass.models.crfemblstm import CrfEmbLstm
from sentclass.models.crflstmlstm import CrfLstmLstm

import sentclass.sentihood as data

device = torch.device("cpu")

lstmfinal = "saves/lstmfinal/lstmfinal-lr0.01-nl2-rnnsz50-dp0.2-va0.8375912408759124-vf0.787758963752766-ta0.8205128205128205-tf0.7801622096241672.pt"
crflstmdiag = "saves/crflstmdiag/crflstmdiag-lr0.01-nl2-rnnsz50-dp0.2-va0.8503649635036497-vf0.7945598964727718-ta0.805318138651472-tf0.7643504515063997.pt"
crfemblstm = "saves/crfemblstm/crfemblstm-lr0.01-nl2-rnnsz50-dp0.2-va0.8686131386861314-vf0.7965845969516062-ta0.8328584995251662-tf0.77851354038455.pt"
crflstmlstm = "saves/crflstmlstm/crflstmlstm-lr0.01-nl2-rnnsz50-dp0.2-va0.864963503649635-vf0.8110252280727556-ta0.8433048433048433-tf0.7990495040192692.pt"

lstmfinal = torch.load(lstmfinal).to(device)
crflstmdiag = torch.load(crflstmdiag).to(device)
crfemblstm = torch.load(crfemblstm).to(device)
crflstmlstm = torch.load(crflstmlstm).to(device)
lstmfinal.eval()
crflstmdiag.eval()
crfemblstm.eval()
crflstmlstm.eval()

TEXT, LOCATION, ASPECT, SENTIMENT = data.make_fields()
train, valid, test = data.SentihoodDataset.splits(
    TEXT, LOCATION, ASPECT, SENTIMENT, flat=True, path="data")

data.build_vocab(TEXT, LOCATION, ASPECT, SENTIMENT, train, valid, test)
TEXT.vocab.load_vectors(vectors=GloVe(name="840B"))
TEXT.vocab.vectors[TEXT.vocab.stoi["transit-location"]] = (
(TEXT.vocab.vectors[TEXT.vocab.stoi["transit"]] +
    TEXT.vocab.vectors[TEXT.vocab.stoi["location"]]) / 2
)

f = lambda num: f"{num:0.2f}"

print("===== Hand-crafted example for paper =====")

t0 = "location1 is a great place to live but not too convenient".split()
x, lens = TEXT.process([t0])
a = torch.LongTensor([[0]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[1]])
s0, psi_ys0 = crfemblstm.observe(x, lens, l, a, y)
ss0 = torch.softmax(s0, -1)
sy0 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok0 = list(zip(["<bos>"] + t0 + ["<eos>"], ss0[0].tolist(), psi_ys0[0,:,1:,1:].transpose(-1,-2).tolist()))
print("Positive sentiment, general")
for w, x, y in ok0:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()

t0 = "location1 is a great place to live but not too convenient".split()
x, lens = TEXT.process([t0])
a = torch.LongTensor([[3]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[2]])
s0, psi_ys0 = crfemblstm.observe(x, lens, l, a, y)
ss0 = torch.softmax(s0, -1)
sy0 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok0 = list(zip(["<bos>"] + t0 + ["<eos>"], ss0[0].tolist(), psi_ys0[0,:,1:,1:].transpose(-1,-2).tolist()))
print("Negative sentiment, transit-location")
for w, x, y in ok0:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()
import pdb; pdb.set_trace()

#t0 = ["she", "liked", "location1", "but", "others", "said", "it", "'s", "rough"]
x, lens = TEXT.process([t0])
a = torch.LongTensor([[0]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[1]])
s0, psi_ys0 = crfemblstm.observe(x, lens, l, a, y)
ss0 = torch.softmax(s0, -1)
sy0 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok0 = list(zip(["<bos>"] + t0 + ["<eos>"], ss0[0].tolist(), psi_ys0[0,:,1:,1:].transpose(-1,-2).tolist()))
print("Positive sentiment")
for w, x, y in ok0:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()

x, lens = TEXT.process([t0])
a = torch.LongTensor([[2]])
y = torch.LongTensor([[2]])
s02, psi_ys02 = crfemblstm.observe(x, lens, l, a, y)
ss02 = torch.softmax(s02, -1)
sy02 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok02 = list(zip(["<bos>"] + t0 + ["<eos>"], ss02[0].tolist(), psi_ys02[0,:,1:,1:].transpose(-1,-2).tolist()))
for w, x, y in ok02:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()

import pdb; pdb.set_trace()
 
print("===== Hand-crafted experiment for negation =====")

t0 = ["location1", "is", "very", "safe"]
x, lens = TEXT.process([t0])
a = torch.LongTensor([[2]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[1]])
s0, psi_ys0 = crfemblstm.observe(x, lens, l, a, y)
ss0 = torch.softmax(s0, -1)
sy0 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok0 = list(zip(["<bos>"] + t0 + ["<eos>"], ss0[0].tolist(), psi_ys0[0,:,1:,1:].transpose(-1,-2).tolist()))
print("Positive sentiment")
for w, x, y in ok0:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()

t0 = ["location1", "is", "very", "safe"]
x, lens = TEXT.process([t0])
y = torch.LongTensor([[2]])
s02, psi_ys02 = crfemblstm.observe(x, lens, l, a, y)
ss02 = torch.softmax(s02, -1)
sy02 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok02 = list(zip(["<bos>"] + t0 + ["<eos>"], ss02[0].tolist(), psi_ys02[0,:,1:,1:].transpose(-1,-2).tolist()))
for w, x, y in ok02:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()

t1 = ["location1", "is", "not", "safe"]
x, lens = TEXT.process([t1])
a = torch.LongTensor([[2]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[2]])
s1, psi_ys1 = crfemblstm.observe(x, lens, l, a, y)
ss1 = torch.softmax(s1, -1)
sy1 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok1 = list(zip(["<bos>"] + t1 + ["<eos>"], ss1[0].tolist(), psi_ys1[0,:,1:,1:].transpose(-1,-2).tolist()))
print("Negative Sentiment")
for w, x, y in ok1:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()

t2 = ["location1", "is", "not", "safe"]
x, lens = TEXT.process([t2])
a = torch.LongTensor([[2]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[1]])
s2, psi_ys2 = crfemblstm.observe(x, lens, l, a, y)
ss2 = torch.softmax(s2, -1)
sy2 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
ok2 = list(zip(["<bos>"] + t2 + ["<eos>"], ss2[0].tolist(), psi_ys2[0,:,1:,1:].transpose(-1,-2).tolist()))
print("Positive sentiment")
for w, x, y in ok2:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t{y}")
print()


t0 = ["location1", "is", "very", "safe"]
x, lens = TEXT.process([t0])
y = torch.LongTensor([[1]])
#s02, psi_ys02 = crfemblstm.observe(x, lens, l, a, y)
s02 = lstmfinal.observe(x, lens, l, a, y)
ss02 = torch.softmax(s02, -1)
#sy02 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
sy02 = torch.softmax(lstmfinal(x, lens, [l,a], [l,a]), -1)
ok02 = list(zip(["<bos>"] + t0 + ["<eos>"], ss02[0].tolist()))
for w, x in ok02:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]")
print()
t0 = ["location1", "is", "not", "safe"]
x, lens = TEXT.process([t0])
y = torch.LongTensor([[2]])
#s02, psi_ys02 = crfemblstm.observe(x, lens, l, a, y)
s02 = lstmfinal.observe(x, lens, l, a, y)
ss02 = torch.softmax(s02, -1)
#sy02 = torch.softmax(crfemblstm(x, lens, [l,a], [l,a]), -1)
sy02 = torch.softmax(lstmfinal(x, lens, [l,a], [l,a]), -1)
ok02 = list(zip(["<bos>"] + t0 + ["<eos>"], ss02[0].tolist()))
for w, x in ok02:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]")
print()
#import pdb; pdb.set_trace()
print("===== Comparisons between models =====")
t2 = ["location1", "is", "not", "safe"]
x, lens = TEXT.process([t2])
a = torch.LongTensor([[2]])
l = torch.LongTensor([[0]])
y = torch.LongTensor([[1]])

s0 = lstmfinal.observe(x, lens, l, a, y)
s1 = crflstmdiag.observe(x, lens, l, a, y)
s2, psi_ys = crfemblstm.observe(x, lens, l, a, y)
s3, psi_ys1 = crflstmlstm.observe(x, lens, l, a, y)
#import pdb; pdb.set_trace()

ss0 = torch.softmax(s0, -1)
ss1 = torch.softmax(s1, -1)
ss2 = torch.softmax(s2, -1)
ss3 = torch.softmax(s3, -1)
print()
print(" ".join(t2))
print(f"aspect: {ASPECT.vocab.itos[a.item()]}")
print(f"location: {LOCATION.vocab.itos[l.item()]}")
print(f"sentiment: {SENTIMENT.vocab.itos[y.item()]}")
ok = list(zip(["<bos>"] + t2 + ["<eos>"], ss0[0].tolist(), ss1[0,:,:-1].tolist(), ss3[0].tolist(), ss2[0].tolist(), psi_ys[0,:,1:,1:].transpose(-1, -2).tolist()))
print("## model comparison ##")
for w, x, y, a, z, p in ok:
    print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t[{' '.join(map(f, y))}]\t[{' '.join(map(f, a))}]\t[{' '.join(map(f, z))}]")
print()

import pdb; pdb.set_trace()
print("===== Comparisons on valid between models =====")
dataset = valid
for i in range(0, len(dataset)):
    example = Batch([dataset[i]], dataset, device)
    if example.sentiments.item() != 0:
        s0 = lstmfinal.observe(example.text[0], example.text[1], example.locations, example.aspects, example.sentiments)
        s1 = crflstmdiag.observe(example.text[0], example.text[1], example.locations, example.aspects, example.sentiments)
        s2, psi_ys = crfemblstm.observe(example.text[0], example.text[1], example.locations, example.aspects, example.sentiments)
        s3, psi_ys1 = crflstmlstm.observe(example.text[0], example.text[1], example.locations, example.aspects, example.sentiments)
        #import pdb; pdb.set_trace()

        ss0 = torch.softmax(s0, -1)
        ss1 = torch.softmax(s1, -1)
        ss2 = torch.softmax(s2, -1)
        ss3 = torch.softmax(s3, -1)
        print()
        print(" ".join(dataset[i].text))
        print(f"aspect: {ASPECT.vocab.itos[example.aspects.item()]}")
        print(f"location: {LOCATION.vocab.itos[example.locations.item()]}")
        print(f"sentiment: {SENTIMENT.vocab.itos[example.sentiments.item()]}")
        ok = list(zip(["<bos>"] + dataset[i].text + ["<eos>"], ss0[0].tolist(), ss1[0,:,:-1].tolist(), ss3[0].tolist(), ss2[0].tolist(), psi_ys[0,:,1:,1:].transpose(-1, -2).tolist()))
        print("## model comparison ##")
        for w, x, y, a, z, p in ok:
            print(f"{w:<10}:\t[{' '.join(map(f, x))}]\t[{' '.join(map(f, y))}]\t[{' '.join(map(f, a))}]\t[{' '.join(map(f, z))}]")
        print()
        #print("## conditional || unary || interaction matrix ##")
        #for w, x, y, a, z, p in ok:
            #print(f"{w:<10}:\t[{' '.join(map(f, z))}]\t{p}")

        import pdb; pdb.set_trace()
