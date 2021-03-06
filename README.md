# SentClass
Sentence classification on the Sentihood dataset with a couple CRF models.

## Requirements
Code is tested with [torch](https://github.com/pytorch/pytorch) 1.0.0, [pyro](https://github.com/uber/pyro) 0.3.0,
[torchtext](https://github.com/pytorch/text) 0.4.0, and [spacy](https://spacy.io) 2.0.18, and nltk 3.4.

## Data
Data was obtained from [jack](https://github.com/uclmr/jack/tree/master/data/sentihood).

## Training
Run the models with the following command:
```
python main.py --flat-data --model {lstmfinal | crflstmdiag | crflstmlstm | crfemblstm} {--save} {--devid #}
```
Add the `--save` option to save checkpoints along with their valid and test performance.
Be sure to create the `save/{modelname}` directory before saving.
Models are saved based on validation accuracy.

Use the `--devid #` option to utilize a GPU.


## Status
* Looks like `crfneg` and `crfnegt` both have the same deficiency: since we use the output of the
BLSTM to parameterize $$b_t$$ which is supposed to control negation, the embeddings learn to only give
saliency whil $$b_t$$ is used to indicate sentiment. Tricky LSTM!
* Next step: Constrain the parameterization of $$\phi(b_t)$$ further. Use BLSTM to choose which word to attend to.
Negation will be a function of the attended word's embedding only.
* Constraint worked, but model is weak and no sign of capturing negation.
