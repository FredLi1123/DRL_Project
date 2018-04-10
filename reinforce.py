# coding: utf-8

import sys
import argparse
import os
import random
import numpy as np


def _get_action(logits, nactions, testing):
    # test greedily and sample according to the policy
    if testing:
        action = np.argmax(logits)
    else:
        action = np.random.choice(len(logits), p=logits)
    
    return action


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.
    def __init__(self, model, lr=1e-3):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.optimizer = keras.optimizers.Adam(lr=lr)  
        self.model.compile(optimizer=self.optimizer, loss=Reinforce.reinforce_loss)

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        """
        states, rewards, total_reward = self.generate_episode(testing=False)
        episode_len = len(rewards)
        
        running_sum = 0.0
        # calculate returns
        for i in range(episode_len-1, -1, -1):
            running_sum = rewards[i] + gamma * running_sum
            returns[i] = running_sum
        
        loss = returns * log(probs)
        return loss
        """
        nactions = env.action_space.n
        # obtain information from an episode
        states, actions, rewards, training_reward = self.generate_episode(env, render=False)
        episode_len = len(rewards)
        rewards = np.array(rewards, dtype='float')
        returns = np.zeros(episode_len)
        embedded_returns = np.zeros((episode_len, nactions))
        running_sum = 0.0
        # calculate returns
        for i in range(episode_len-1, -1, -1):
            running_sum = rewards[i] + gamma * running_sum
            returns[i] = running_sum
        # standardize returns
        returns = (returns - np.mean(returns)) / np.std(returns)
        # embed the returns with corresponding actions
        for i in range(episode_len):
            embedded_returns[i][actions[i]] = returns[i]
        # train the model
        states = np.concatenate(states).reshape(episode_len, -1)
        self.model.fit(states, embedded_returns, batch_size=episode_len, shuffle=True, epochs=1, verbose=1)
        # apply weight clipping
        self.clip_weights(-1.0, 1.0)
        

    def evaluate(self, env, test_episodes, render=False):
        # generate episodes and obtain mean and standard deviation of rewards
        records = np.zeros(test_episodes)
        for i in range(test_episodes):
            _, _, _, reward = self.generate_episode(env, render=render, testing=True)
            records[i] = reward
        
        return np.mean(records), np.std(records)
    
    def generate_episode(self, testing=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        is_terminal = False

        """
        :param model - a LSTM-based word language model
        :param sentence - ground truth
        
        initialize h = 0, c = 0, x = w_0
        for t = 1: len(sentence)
            sample (output, h_next, c_next) based on Gaussian policy
            reward = crossentropy(w_t, output) = Pr(output[w_t])
            total_reward += reward
            states.append(h, c, x)
            rewards.append(reward)

            h = h_next
            c = c_next
            x = w_t
        
        return probs, rewards, total_reward
        """

        while not is_terminal:
            # predict and obtain the next step
            logits = self.model.predict(cur_state.reshape(1, -1), batch_size=1)
            action = _get_action(logits[0], nactions, testing=testing)
            nextstate, reward, is_terminal, _ = env.step(action=action)
            # record the information
            states.append(cur_state)
            actions.append(action)
            rewards.append(reward/100)
            # get to the next step
            cur_state = nextstate
        
        total_reward = sum(rewards)*100
        
        return states, actions, rewards, total_reward
    
    def sample(h, c, x):
        """
        output, h_next, c_next = self.model(h, c, x)
        output_sample, h_next_sample, c_next_sample = torch.random.normal(mu = (output, h_next, c_next), self.Sigma)
        probs = torch.distributions.Gaussian(output_sample, mu=output, sigma=self.sigma)

        return probs, output_sample
        """

    def clip_weights(self, lb, ub):
        # fetch current weights
        weights = self.model.get_weights()
        # clip the weights in each layer
        for w in weights:
            w = np.minimum(ub, np.maximum(lb, w))
        # reset the weights
        self.model.set_weights(weights)
    
    def reset_learning_rate(self, lr):
        K.set_value(self.optimizer.lr, lr)

    @staticmethod
    def reinforce_loss(y_true, y_pred):
        # add a smaller number to the loss to prevent explosion (nan)
        loss = -K.sum(y_true * K.log(y_pred + 1e-7), axis=1)
        return K.sum(loss)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="The learning rate.")
    parser.add_argument('--model-file', dest='model_file', type=str, 
                        default=None, help="The saved model")

    return parser.parse_args()


def main(args):
    # create configurations
    cfg = configurations.generate_reinforce_config()
    # set seeds
    np.random.seed(cfg['numpy_seed'])
    random.seed(cfg['rand_seed'])
    K.tf.set_random_seed(cfg['tf_seed'])

    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    model_file = args.model_file
    lr = args.lr

    # parse configurations
    train_report_interval = cfg['train_report_interval']
    test_report_interval = cfg['test_report_interval']
    test_episode = cfg['test_episode']
    converge_lr = cfg['converge_lr']

    # Create the environment and set environment seed
    env = gym.make('LunarLander-v2')
    env.seed(cfg['env_seed'])
    
    
    if model_file is None:
        # if model not saved, oad the policy model from json
        f = open(model_config_path, 'r')
        model = keras.models.model_from_json(f.read())
        f.close()
    else:
        # load the saved model
        model = keras.models.load_model(args.model_file, 
                                        custom_objects={'reinforce_loss': Reinforce.reinforce_loss})

    # TODO: Train the model using REINFORCE.
    reinforce_model = Reinforce(model, lr)
    total_training_reward = 0.0
    # performance recorders
    records = np.zeros((num_episodes // test_report_interval, 2))
    logfile = open('results.txt', mode='w')

    for episode in range(num_episodes):
        """
        loss = reinforce_model.train()
        loss.backward()

        optimizer.step()
        """
        training_reward = reinforce_model.train(env)
        total_training_reward += training_reward

        if (episode+1) % train_report_interval == 0:
            # record training rewards
            logfile.write('episode: '+str(episode+1)+'\n')
            logfile.write('training reward: '+str(total_training_reward / TRAIN_REPORT_INTERVAL)+'\n')
            logfile.flush()
            total_training_reward = 0.0

        if (episode+1) % test_report_interval == 0:
            # record testing rewards and save model
            mu, sigma = reinforce_model.evaluate(env, test_episode)
            print('episode:', episode+1)
            print('mean reward:', mu)
            print('standard deviation:', sigma)
            logfile.write('mean reward: '+str(mu)+'\n')
            logfile.write('standard deviation: '+str(sigma)+'\n')
            logfile.flush()
            records[episode // TEST_REPORT_INTERVAL, 0] = mu
            records[episode // TEST_REPORT_INTERVAL, 1] = sigma
            reinforce_model.model.save('mlp_model.h5')

            # anneal the learning rate if we have reached the target
            if mu > 210:
                reinforce_model.reset_learning_rate(converge_lr)
    
    np.save('results.npy', records)
    logfile.close()

if __name__ == '__main__':
    main(sys.argv)
