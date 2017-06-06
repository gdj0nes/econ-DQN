from __future__ import division

import os
import numpy as np
import tensorflow as tf
import tqdm

from buffers import ExperienceBuffer
from agents import DQN
from environments import LifeCycle


class LifeTimeModel(object):
    """Train a life cycle model
    
    """

    def __init__(self, name, learning_rate=1e-4, beta=0.935, batch_size=128):

        self.name = name
        self.log_path = os.path.join('logs', name)
        print self.log_path

        # Network parameters start here
        self.delta = 1e-4
        self.batch_size = batch_size
        self.beta = beta
        self.episode_size = 256

        # Environment
        self.action_dim = 20
        self.life_span = 35
        self.R = 1.03
        self.gamma = 1.5
        self.env = LifeCycle(self.gamma, self.R, self.life_span, self.episode_size, self.action_dim)
        self.state_dim = self.env.state_size
        self.episodes = int(9e5)
        self.eval_freq = 1000
        self.e = .8

        # Experiment Buffer
        self.buffer_size = self.episode_size * 100000  # TODO: beware changing that
        self.buffer = ExperienceBuffer(self.buffer_size, self.batch_size, self.state_dim)

        # Model
        self.pre_fill = 100 * self.batch_size
        self.update_freq = self.episode_size * 10  # TODO: this needs to be multiple of episode size
        self.save_freq = self.update_freq * 10
        self.sess = tf.Session()
        actGrid = self.env.actGrid
        self.DQN = DQN(self.sess, self.log_path, learning_rate, self.action_dim, self.state_dim, beta, actGrid)
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self):

        DQN = self.DQN
        # Load existing model
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(DQN.test_phase)  # For batch normalization
        total_steps = 0
        for episode in tqdm.trange(self.episodes):
            s, term = self.env.reset()
            cum_rewards = np.zeros(self.episode_size)
            t = 0
            while term:
                if self.e > np.random.rand():
                    action = np.random.randint(0, self.action_dim, size=self.episode_size)
                else:
                    action = self.sess.run(DQN.predict, {DQN.state: s})
                    # rand = np.random.randint(-2, 3)
                    # if rand + action > self.action_dim:
                    #     action = action - (rand % action)
                    # elif rand + action < self.action_dim:
                    #     action = abs(rand)
                    # else:
                    #     action += rand

                reward, s1, term = self.env.step(action)
                reward[reward < -20] = -20
                cum_rewards += reward * self.beta ** t
                t += 1
                self.buffer.add_batch(action, reward, s1, term)
                s = s1
                total_steps += self.episode_size
                if (self.buffer.count > self.pre_fill) and (total_steps % self.update_freq == 0):
                    batch = self.buffer.sample()
                    DQN.update_model(batch)
                    self.e = max(self.e * 0.999, 0.1)
                    if total_steps % self.save_freq == 0:
                        self.saver.save(self.sess, self.log_path)

            if episode % 25 == 0:
                summary = self.sess.run(DQN.episode_summary_op, feed_dict={DQN.rewards_episode: cum_rewards})
                DQN.writer.add_summary(summary, episode)

    def load(self):
        self.saver.restore(self.sess, self.log_path)
