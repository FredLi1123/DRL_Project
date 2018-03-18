# coding: utf-8
import argparse
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model Generation')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--generator_size', type=int, default=10,
                    help='where generation starts')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = args.batch_size
test_data = batchify(corpus.test, eval_batch_size)


def get_gen_batch(source, i, evaluation=False):
    if args.bptt <= len(source)-1-i:
        gen_data = Variable(source[i:i+args.generator_size], volatile=evaluation)
        gen_target = Variable(source[i+args.generator_size:i+args.bptt].view(-1))
        return gen_data, gen_target
    else:
        return None, None


def generate(data):
    model.eval()
    ntokens = len(corpus.dictionary)
    generated = torch.zeros((args.bptt-args.generator_size, eval_batch_size, ntokens)).float()
    if args.cuda:
        generated = generated.cuda()
    inputs = data
    hidden = model.init_hidden(eval_batch_size)
    
    for i in range(args.bptt-args.generator_size):
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)
        outputs = outputs[-1]
        generated[i] = outputs.data
        word_weights = outputs.data.div(args.temperature).exp()
        inputs = Variable(torch.multinomial(word_weights, 1).view(1, -1))
    
    return generated


def gen_evaluate(data_source):
    model.eval()
    total_loss = 0.0
    total_len = 0
    ntokens = len(corpus.dictionary)
    
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_gen_batch(data_source, i, evaluation=True)
        if data is not None:
            gen_data = generate(data)
            gen_data_flat = Variable(gen_data.view(-1, ntokens))
            total_loss += len(gen_data) * criterion(gen_data_flat, targets).data
            total_len += len(gen_data)
    
    return total_loss / total_len

generation_loss = gen_evaluate(test_data)
print('=' * 89)
print('| End of generation | generation loss {:5.2f} | generation ppl {:8.2f}'.format(float(generation_loss.cpu()), float(math.exp(generation_loss.cpu()))))
print('=' * 89)