import os
from io import open
import time
import copy

import onnx
from tvm import relay

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition
class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden


def init_hidden(lstm_model, bsz):
    # get the weight tensor and create hidden layer in the same device
    weight = lstm_model.encoder.weight
    # get weight from quantized model
    if not isinstance(weight, torch.Tensor):
        weight = weight()
    device = weight.device
    nlayers = lstm_model.rnn.num_layers
    nhid = lstm_model.rnn.hidden_size
    return (torch.zeros(nlayers, bsz, nhid, device=device),
            torch.zeros(nlayers, bsz, nhid, device=device))


# Load Text Data
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

ntokens = len(corpus.dictionary)

# Load Pretrained Model
model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model.eval()
print(model)

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)

# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = init_hidden(model_, eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig

# Full docs for supported qconfig for floating point modules/ops can be found in docs for quantization (TODO: link)
# Full docs for qconfig_dict can be found in the documents of prepare_fx (TODO: link)
qconfig_dict = {
    "object_type": [
        (nn.Embedding, float_qparams_weight_only_qconfig),
        (nn.LSTM, default_dynamic_qconfig),
        (nn.Linear, default_dynamic_qconfig)
    ]
}
# Deepcopying the original model because quantization api changes the model inplace and we want
# to keep the original model for future comparison
model_to_quantize = copy.deepcopy(model)
input_fp32 = torch.randn(1, 3, 224, 224)
prepared_model = prepare_fx(model_to_quantize, qconfig_dict, (input_fp32))
print("prepared model:", prepared_model)
quantized_model = convert_fx(prepared_model)
print("quantized model", quantized_model)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

# torch.set_num_threads(1)

# def time_model_evaluation(model, test_data):
#     s = time.time()
#     loss = evaluate(model, test_data)
#     elapsed = time.time() - s
#     print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

# time_model_evaluation(model, test_data)
# time_model_evaluation(quantized_model, test_data)

input_fp32 = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.trace(quantized_model.eval(), input_fp32)
torch.jit.save(scripted_model, "./data/fx_ptdq.torchscript")
torch.onnx.export(quantized_model, input_fp32, "./data/fx_ptdq.onnx", verbose=False)

shape_list = [("input", input_fp32.numpy().shape)]
torchscript_model = torch.jit.load("./data/fx_ptdq.torchscript")
mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)
print(mod)
print("*************Print Pytorch IR Success************************")

onnx_model = onnx.load("./data/fx_ptdq.onnx")
mod, params = relay.frontend.from_onnx(
    onnx_model, {"x": input_fp32.numpy().shape}, {"x": "float32"}
)
print(mod)
print("*************Print Onnx IR Success************************")