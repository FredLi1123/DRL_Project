# coding: utf-8
# first party
import torch
import torch.nn as nn
from torch.autograd import Variable


class GaussianNet(nn.Module):

    def __init__(self, sigma):
        super(GaussianNet, self).__init__()
        self.sigma = sigma
    
    def forward(self, mu, sample):
        logprob = -1/(2*self.sigma*self.sigma) * torch.sum((sample-mu)**2)
        return logprob


class Reinforce(nn.Module):
    # Implementation of the policy gradient method REINFORCE.
    def __init__(self, policy, sigma=0.1, gamma=1.0):
        super(Reinforce, self).__init__()
        self.policy = policy
        self.sigma = sigma
        self.gamma = gamma
        self.sampler = GaussianNet(self.sigma)

    def forward(self, inputs, targets, hidden):
        '''
        :param inputs: seq_len, 1
        :param targets: seq_len
        :return:
        '''

        probs, rewards, hidden = self.generate_episode(inputs, targets, hidden)
        episode_len = len(rewards)
        print(probs)
        print(rewards)

        running_sum = 0.0
        returns = Variable(torch.zeros(episode_len), requires_grad=False).cuda()
        # calculate returns
        for i in range(episode_len-1, -1, -1):
            running_sum = rewards[i] + self.gamma * running_sum
            returns[i] = running_sum

        print(returns)

        loss = returns * torch.log(probs)
        total_loss = -torch.mean(loss)
        return total_loss, hidden

    def generate_episode(self, inputs, targets, hidden):
        '''
        :param inputs: seq_len, 1
        :param targets: seq_len
        '''
        probs = []
        rewards = []

        len_sentence = inputs.size(0)

        loss_func = nn.CrossEntropyLoss()

        hc = torch.cat(hidden, dim=2)
        std = torch.zeros(hc.size()).fill_(self.sigma).cuda()

        for t in range(len_sentence):
            print(t)
            output, hidden = self.policy(inputs[t].unsqueeze(dim=0), hidden)
            hc = torch.cat(hidden, dim=2)  # 1, 1, hidden_size * 2
            # execute policy: sample from mean hidden, std sigma

            next_hc = Variable(torch.normal(hc.data, std), requires_grad=False).cuda()
            # calculate log-prob and return as Variable
            logprob = self.sampler(hc, next_hc)
            reward = loss_func(output.squeeze(dim=1), targets[t]).data

            probs.append(logprob)
            rewards.append(reward)

            h = hc[:, :, :hc.size(2) // 2].contiguous()
            c = hc[:, :, hc.size(2) // 2:].contiguous()
            hidden = (h, c)

        probs = torch.cat(probs)
        return probs, rewards, hidden






