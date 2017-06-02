import os
import tensorflow as tf
import numpy as np


class ActorCritic(object):

    def __init__(self, log_path, state_dim, action_dim, learning_rate, batch_size,
                 beta=0.975, tau=0.005, delta=1e-6, sess=None):

        self.batch_size = batch_size
        self.sess = sess

        # Environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta

        # Optimization
        self.train_steps = 0
        self.max_gradient = 5
        self.reg_param = None
        self.tau = tau
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step')
        self.incr_GS = tf.assign_add(self.global_step, 1)
        # Phase for batch normalization
        self.learn_phase = tf.Variable(True, name='learning_phase')
        self.train_phase = tf.assign(self.learn_phase, True)
        self.test_phase = tf.assign(self.learn_phase, False)

        # Decay the mean over time
        self.noise_std = tf.Variable(.6, dtype=tf.float32)  # Want to decay this value
        self.step_rate = 300
        self.noise_decay = .9995
        # TODO: more intelligent noise
        self.noise_decay_op = tf.assign(self.noise_std, tf.maximum(self.noise_std * self.noise_decay, 0.6))
        self.noise_var = tf.random_normal(shape=[self.batch_size, self.action_dim], mean=0, stddev=self.noise_std, name='action_noise')
        self.delta = delta

        self.build_model()
        # Summary Params
        self.sum_freq = 20
        self.writer = tf.summary.FileWriter(log_path, sess.graph)

    # noinspection PyAttributeOutsideInit
    def build_model(self):
        # Exploration noise

        # Initialize Model Inputs
        with tf.variable_scope('inputs'):
            self.state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='state')
            self.action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='action')

        # Define Q-Network forward pass
        with tf.name_scope('predict_actions'):
            with tf.variable_scope('actor'):
                self.policy_output = self.build_actor(self.state)

            with tf.variable_scope('critic'):
                self.value_output = self.build_critic(self.state, self.action)
                self.action_gradient = tf.gradients(self.value_output, self.action)[0]


        self.predicted_actions = tf.identity(self.policy_output, name="pred_actions")  # Why is this necessary
        tf.summary.histogram("savings_rate", self.predicted_actions)
        tf.summary.histogram("action_scores", self.value_output)
        # Get the variables in the target network
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")

        with tf.name_scope("reward_estimation"):
            # These are filled with values coming from the queue
            self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")
            self.terminal = tf.placeholder(tf.float32, (None,), name="next_state_masks")
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

            # Construct the target networks
            with tf.variable_scope("target_actor"):
                self.target_policy_output = self.build_actor(self.next_states)
            with tf.variable_scope("target_critic"):
                self.target_value_output = self.build_critic(self.next_states, self.target_policy_output)

            self.next_action_scores = tf.stop_gradient(self.target_value_output)[:, 0] * self.terminal
            self.future_rewards = self.rewards + self.beta * self.next_action_scores

            tf.summary.histogram("next_action_scores", self.next_action_scores)

        with tf.name_scope("compute_gradients"):

            # compute gradients for critic network
            self.mse = tf.reduce_mean(tf.squared_difference(self.value_output[:, 0], self.future_rewards))
            if self.reg_param:
                self.critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
                self.critic_loss = self.mse + self.reg_param * self.critic_reg_loss
            else:
                self.critic_loss = self.mse

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):  #
                self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_vars)

                self.q_action_grad = tf.placeholder(tf.float32, (None, self.action_dim), name="q_action_grad")
                actor_policy_gradients = tf.gradients(self.policy_output, actor_vars, -self.q_action_grad)
                # This should be a one liner
                self.actor_gradients = list(zip(actor_policy_gradients, actor_vars))

                # collect all gradients
                self.gradients = self.actor_gradients + self.critic_gradients
                if self.max_gradient:
                    self.gradients = [(tf.clip_by_norm(grad, self.max_gradient), var) for grad, var in self.gradients]
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

        tf.summary.scalar("critic_loss", self.critic_loss)
        # update target network with Q network
        # TODO: Tensorflow exponential moving avergage
        with tf.name_scope("update_target_network"):
            self.target_update = []
            # Target Actor
            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
            target_actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_actor")
            for v_source, v_target in zip(actor_vars, target_actor_variables):
                update_op = v_target.assign_sub(self.tau * (v_target - v_source))
                # update_op = v_target.assign_sub(v_source)
                self.target_update.append(update_op)
            # Target Critic
            critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
            target_critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")
            for v_source, v_target in zip(critic_vars, target_critic_variables):
                # update_op = v_target.assign_sub(v_source)
                update_op = v_target.assign_sub(self.tau * (v_target - v_source))
                self.target_update.append(update_op)

            # Group all assignment operations together
            self.target_update = tf.group(*self.target_update)

        tf.summary.scalar('noise_std', self.noise_std)
        self.summary_op = tf.summary.merge_all()


    def updateModel(self, sess, batch_data):

        sess.run(self.train_phase)
        a_batch, r_batch, pre_s, post_s, t_batch = batch_data
        policy_outs = sess.run(self.policy_output, {self.state: pre_s})
        action_grads = sess.run(self.action_gradient, feed_dict={self.state: pre_s, self.action: policy_outs})
        if self.train_steps % self.sum_freq == 0:
            summary, _ = sess.run([self.summary_op, self.train_op],
                                  feed_dict={self.state: pre_s,
                                             self.next_states: post_s,
                                             self.terminal: t_batch,
                                             self.action: a_batch,
                                             self.rewards: r_batch,
                                             self.q_action_grad: action_grads})
            self.writer.add_summary(summary, self.train_steps)
        else:
            _ = sess.run([self.train_op], feed_dict={self.state: pre_s,
                                                     self.next_states: post_s,
                                                     self.terminal: t_batch,
                                                     self.action: a_batch,
                                                     self.rewards: r_batch,
                                                     self.q_action_grad: action_grads})

        self.train_steps += 1
        sess.run(self.noise_decay_op)
        sess.run(self.target_update)
        sess.run([self.incr_GS, self.test_phase])

    def build_actor(self, state):

        H1 = 50
        H2 = 50
        H3 = 50

        # FC1
        x = tf.layers.dense(state, H1, name='FC1')
        x = tf.layers.batch_normalization(x, training=self.learn_phase)
        x = tf.nn.elu(x)
        # FC2
        x = tf.layers.dense(x, H2, name='FC2')
        # x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.elu(x)
        # FC 3
        # x = tf.layers.dense(x, H3, name='FC3')
        # # x = tf.layers.batch_normalization(x, training=True)
        # x = tf.nn.elu(x)

        action = tf.layers.dense(x, self.action_dim)
        action = tf.nn.sigmoid(action)

        return action

    def build_critic(self, state, action):

        H0 = 15
        H1 = 75
        H2 = 75
        H3 = 50

        # State branch
        sX = tf.layers.dense(state, H0, name='S1')
        sX = tf.layers.batch_normalization(sX, training=self.learn_phase)
        sX = tf.nn.relu(sX)
        sX = tf.layers.dense(sX, H1, name='S2')
        sX = tf.nn.elu(sX)

        # Action branch
        aX = tf.layers.dense(action, H0, name='A1')
        aX = tf.layers.batch_normalization(aX, training=self.learn_phase)
        aX = tf.nn.relu(aX)
        aX = tf.layers.dense(aX, H1, name='A2')
        aX = tf.nn.elu(aX)

        # Merge the two branches
        merge = tf.concat([aX, sX], axis=1, name='Merge')
        x = tf.layers.dense(merge, H2)
        x = tf.nn.elu(x)
        x = tf.layers.dense(x, H3)
        x = tf.nn.elu(x)
        value = tf.layers.dense(x, self.action_dim)

        return value

    def sampleAction(self, state, step, search=True):

        policy_outs, noise = self.sess.run([self.policy_output, self.noise_var], feed_dict={self.state: state})

        if search:
            policy_outs += noise / np.log(step + np.e)
            policy_outs[policy_outs < 0] = np.abs(policy_outs[policy_outs < 0])
            policy_outs[policy_outs > 1] = 1 - (policy_outs[policy_outs > 1] - 1)
            policy_outs = np.clip(policy_outs, self.delta, 1 - self.delta)

        return policy_outs