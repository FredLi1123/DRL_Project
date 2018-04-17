from torch.autograd import Variable
import torch.nn as nn
import argparse
import data
import model
import torch
import reinforce
from tqdm import tqdm
from torch import optim
import time


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, cfg):
    seq_len = min(cfg['max_len'], len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], requires_grad=False)
    target = Variable(source[i+1:i+1+seq_len].view(-1), requires_grad=False)
    return data, target


def evaluate(data_source, model, cfg):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = cfg['dict_size']
    hidden = model.init_hidden(cfg['batch_size'])
    criterion = nn.CrossEntropyLoss()
    for i in range(0, data_source.size(0) - 1, cfg['max_len']):
        data, targets = get_batch(data_source, i, cfg)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    model.train()
    return total_loss[0] / len(data_source)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda()
    return data


def save_model(path, model):
    with open(path, 'wb') as f:
        torch.save(model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=200,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--gpu', action='store_true',
                        help='use GPU')
    parser.add_argument('--init', type=str,
                        default='model_200.pt', help="The LSTM model")

    args = parser.parse_args()

    corpus = data.Corpus(args.data)
    cfg = dict()
    cfg['dict_size'] = len(corpus.dictionary)
    cfg['init'] = args.init
    cfg['max_len'] = args.bptt
    cfg['epochs'] = args.epochs
    cfg['GPU'] = args.gpu
    cfg['lr'] = args.lr
    cfg['batch_size'] = args.batch_size
    cfg['saveto'] = './'
    train_data = batchify(corpus.train, 1)
    val_data = batchify(corpus.valid, cfg['batch_size'])
    test_data = batchify(corpus.test, cfg['batch_size'])

    with open(cfg['init'], 'rb') as f:
        policy = torch.load(f)

    reinforce_model = reinforce.Reinforce(policy=policy)

    ntokens = cfg['dict_size']
    total_loss = 0.0

    optimizer = optim.Adam(reinforce_model.parameters(), lr=cfg['lr'])
    start_time = time.time()
    for epoch in range(cfg['epochs']):
        hidden = policy.init_hidden(bsz=1)
        for i in tqdm(range(0, cfg['max_len'], cfg['max_len'])):
            optimizer.zero_grad()
            data, targets = get_batch(train_data, i, cfg)
            loss, hidden = reinforce_model(data, targets, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += loss.data
            loss.backward()
            optimizer.step()
            # if (i // cfg['max_len'] + 1) % 50 == 0:
            #     print(total_loss / 50)
            #     total_loss = 0.0

        print('Epoch: ', epoch, ' elapse:', time.time() - start_time)

        loss = evaluate(val_data, reinforce_model.policy, cfg)

        save_path = cfg['saveto'] + '_epoch' + str(epoch) + '_loss' + str(loss)
        save_model(save_path, model)
        print('Epoch: ', epoch, ' save to ', save_path)
