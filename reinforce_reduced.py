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
        '''
        :param mu: 1, bsz, 1300
        :param sample: 1, bsz, 1300
        :return:
        '''
        # (sample - mu)**2: 1, bsz, 1300

        logprob = -torch.sum((sample-mu)**2, dim=2)  # 1, bsz
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

    def forward(self, inputs, targets, hidden, base_hidden, alpha):
        '''
        :param inputs: seq_len, bsz
        :param targets: seq_len, bsz
        :return:
        '''

        logprobs, base_rewards, rewards = self.generate_episode(inputs, targets, hidden, base_hidden)
        episode_len = rewards.size(0)
        bsz = rewards.size(1)

        running_sum = torch.zeros((bsz,)).cuda()
        # base_running_sum = torch.zeros((bsz,)).cuda()
        returns = Variable(torch.zeros((episode_len, bsz)), requires_grad=False).cuda()
        # base_returns = Variable(torch.zeros((episode_len, bsz)), requires_grad=False).cuda()
        # calculate returns
        for i in range(episode_len-1, -1, -1):
            running_sum = rewards[i, :] + self.gamma * running_sum
            returns[i, :] = running_sum
            # base_running_sum = base_rewards.data[i, :] + self.gamma * base_running_sum
            # base_returns[i, :] = base_running_sum

        reinforce_loss = torch.mean(returns*logprobs)  # seq_len, bsz
        total_loss = alpha * reinforce_loss + (1-alpha) * base_rewards.mean()

        return total_loss, rewards.mean()

    def generate_episode(self, inputs, targets, hidden, base_hidden):
        '''
        :param inputs: seq_len, bsz
        :param targets: seq_len, bsz
        :param hidden: (h, c); h: 1, bsz, 650; c: 1, bsz, 650
        '''
        probs = []
        outputs = []

        base_scores, _ = self.policy(inputs, base_hidden)

        len_sentence = inputs.size(0)
        hc = torch.cat(hidden, dim=2)  # 1, bsz, 1300
        std = torch.zeros(hc.size()).fill_(self.sigma).cuda()
        # emb: seq_len, bsz, embed_size
        emb = self.policy.drop(self.policy.encoder(inputs))
        
        for t in range(len_sentence):
            # output: 1, bsz, hidden_size
            output, hidden = self.policy.rnn(emb[t, :].unsqueeze(dim=0), hidden)
            # 1, bsz, hidden_size * 2
            hc = torch.cat(hidden, dim=2)  
            # execute policy: sample from mean hidden, std sigma
            next_hc = Variable(torch.normal(hc.data, std), requires_grad=False).cuda()
            # calculate log-prob and return as Variable
            logprob = self.sampler(hc, next_hc)  # 1, bsz
            
            outputs.append(output)
            probs.append(logprob)

            h = next_hc[:, :, :next_hc.size(2) // 2].contiguous()
            c = next_hc[:, :, next_hc.size(2) // 2:].contiguous()
            hidden = (h, c)

        # probs: seq_len, bsz
        probs = torch.cat(probs, dim=0)  
        outputs = torch.cat(outputs, dim=0)
        scores = self.policy.decoder(self.policy.drop(outputs))
        
        # rewards: seq_len, bsz
        rewards = self.loss_func(scores.view(-1, scores.size(2)), targets.view(-1)).data.view(len_sentence, -1)
        base_rewards = self.loss_func(base_scores.view(-1, base_scores.size(2)), targets.view(-1)).view(len_sentence, -1)
            
        return probs, base_rewards, rewards
