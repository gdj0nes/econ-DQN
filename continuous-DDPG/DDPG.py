from __future__ import division

import os
import numpy as np
import tensorflow as tf
import tqdm

from buffers import ExperienceBuffer
from agents import ActorCritic
from environments import LifeCycle#, InfiniteHorizonBatch


class LifeTimeModel(object):
    """Train a life cycle model
    
    """

    def __init__(self, name, actor_eta=2e-4, beta=0.935, batch_size=64):

        self.name = name
        self.log_path = os.path.join('logs', name)
        print self.log_path
        # Network parameters start here
        self.delta = 1e-4
        self.batch_size = batch_size

        # Environment
        self.life_span = 15
        self.R = 1.03
        self.beta = beta
        self.gamma = 1.5
        self.env = LifeCycle(self.gamma, self.R, self.life_span, self.batch_size)
        self.state_dim = self.env.state_size
        self.action_dim = 1
        self.episodes = int(6e5)
        self.eval_freq = 1000

        # Experiment Buffer
        self.buffer_size = self.batch_size * 50000
        self.buffer = ExperienceBuffer(self.buffer_size, self.batch_size, self.state_dim, history_length=1)

        # Models
        self.load_model = False
        self.pre_fill = 1000 * self.batch_size
        self.total_steps = 0
        self.update_freq = batch_size * 20
        self.save_freq = int(self.update_freq * 2)
        self.sess = tf.Session()
        self.actorCritic = ActorCritic(log_path=self.log_path, state_dim=self.state_dim, action_dim=self.action_dim,
                                       learning_rate=actor_eta, batch_size=self.batch_size, beta=beta, delta=self.delta,
                                       sess=self.sess)
        self.saver = tf.train.Saver()

    def train(self):

        # Load existing model
        if self.load_model:
            self.load()
        else:
            self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.actorCritic.test_phase)  # For batch normalization
        for episode in tqdm.trange(self.episodes):
            s, term = self.env.reset()
            age = 0
            while term:
                age += 1
                a = self.actorCritic.sampleAction(s, age)
                r, s1, term = self.env.step(a)
                self.buffer.add_batch(a, r, s1, term)
                s = s1
                self.total_steps += 1

                if (self.buffer.count > self.pre_fill) and (self.total_steps % self.update_freq == 0):
                    batch = self.buffer.sample()
                    self.actorCritic.updateModel(self.sess, batch)

                    if self.total_steps % self.save_freq == 0:
                        self.saver.save(self.sess, self.log_path)

    def load(self):
        self.saver.restore(self.sess, self.log_dir)
