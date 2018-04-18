# coding: utf-8
# first party
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import query_gpu


class GaussianNet(nn.Module):

    def __init__(self, sigma):
        super(GaussianNet, self).__init__()
        self.sigma = sigma
    
    def forward(self, mu, sample):
        '''
        :param mu: 1, bsz, 1300
        :param sample: 1, bsz, 1300
        :return:
        '''
        # (sample - mu)**2: 1, bsz, 1300

        logprob = -1/(2*self.sigma*self.sigma) * torch.sum((sample-mu)**2, dim=2)  # 2, bsz
        return logprob.sum(dim=0, keepdim=True)


class Reinforce(nn.Module):
    # Implementation of the policy gradient method REINFORCE.
    def __init__(self, policy, sigma=1, gamma=1.0):
        super(Reinforce, self).__init__()
        self.policy = policy
        self.sigma = sigma
        self.gamma = gamma
        self.sampler = GaussianNet(self.sigma)
        self.loss_func = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets, hidden):
        '''
        :param inputs: seq_len, bsz
        :param targets: seq_len, bsz
        :return:
        '''

        logprobs, rewards, hidden = self.generate_episode(inputs, targets, hidden)
        episode_len = rewards.size(0)
        bsz = rewards.size(1)

        running_sum = torch.zeros((bsz,)).cuda()
        returns = Variable(torch.zeros((episode_len, bsz)), requires_grad=False).cuda()
        # calculate returns
        for i in range(episode_len-1, -1, -1):
            running_sum = rewards[i, :] + self.gamma * running_sum
            returns[i, :] = running_sum

        loss = returns * logprobs  # seq_len, bsz
        total_loss = -torch.mean(loss)

        return total_loss, hidden, rewards.mean()

    def generate_episode(self, inputs, targets, hidden):
        '''
        :param inputs: seq_len, bsz
        :param targets: seq_len, bsz
        :param hidden: (h, c); h: 1, bsz, 650; c: 1, bsz, 650
        '''
        probs = []
        rewards = []

        len_sentence = inputs.size(0)
        bsz = inputs.size(1)
        hc = torch.cat(hidden, dim=2)  # 1, bsz, 1300

        std = torch.zeros(hc.size()).fill_(self.sigma).cuda()

        for t in range(len_sentence):
            output, hidden = self.policy(inputs[t, :].unsqueeze(dim=0), hidden)
            hc = torch.cat(hidden, dim=2)  # 1, bsz, hidden_size * 2
            # execute policy: sample from mean hidden, std sigma

            next_hc = Variable(torch.normal(hc.data, std), requires_grad=False).cuda()
            # calculate log-prob and return as Variable
            logprob = self.sampler(hc, next_hc)  # 1, bsz
            reward = self.loss_func(output.squeeze(dim=0), targets[t, :]).data.unsqueeze(dim=0)  # 1, bsz
            probs.append(logprob)
            rewards.append(reward)

            h = hc[:, :, :hc.size(2) // 2].contiguous()
            c = hc[:, :, hc.size(2) // 2:].contiguous()
            hidden = (h, c)

        probs = torch.cat(probs, dim=0)  # seq_len, bsz
        rewards = torch.cat(rewards, dim=0)  # seq_len, bsz

        return probs, rewards, hidden






