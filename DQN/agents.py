import tensorflow as tf


class DQN(object):

    def __init__(self, sess, logdir, learning_rate, action_dim, state_dim, beta, act_grid, tau=0.001):
        self.sess = sess or tf.get_default_session()
        self.learning_rate = learning_rate
        self.sum_freq = 20

        self.state_dim = state_dim
        self.state = tf.placeholder(dtype=tf.float32, shape=[state_dim, ])
        self.action_dim = action_dim
        self.act_grid = act_grid
        self.beta = beta
        self.tau = tau
        self.max_grad = 1.

        self.training_step = 0
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.incr_gs = tf.assign_add(self.global_step, 1)
        # Batch-norm stuff
        self.learn_phase = tf.Variable(True, name='learning_phase', trainable=False)
        self.train_phase = tf.assign(self.learn_phase, True)
        self.test_phase = tf.assign(self.learn_phase, False)

        self.xavier = tf.contrib.layers.xavier_initializer()
        # Construct network
        self.build_agent()
        self.writer = tf.summary.FileWriter(logdir, sess.graph)

    # noinspection PyAttributeOutsideInit
    def build_network(self, state):

        H0 = 100
        H1 = 200

        with tf.variable_scope('start'):
            x = tf.layers.dense(state, H0, name='M1', kernel_initializer=self.xavier)
            x = tf.layers.batch_normalization(x, training=self.learn_phase)
            x = tf.nn.elu(x)
            x = tf.layers.dense(x, H1, name='M2', kernel_initializer=self.xavier)
            out = tf.nn.elu(x)

        # Value branch
        with tf.variable_scope('Value'):
            vX = tf.layers.dense(out, H1, name='FC1', kernel_initializer=self.xavier)
            vX = tf.nn.elu(vX)
            value = tf.layers.dense(vX, 1, name='FC2', activation=None, kernel_initializer=self.xavier)

        # Advantage branch
        with tf.variable_scope('Advantage'):
            aX = tf.layers.dense(out, H1, name='FC1')
            aX = tf.nn.elu(aX)
            advantage = tf.layers.dense(aX, self.action_dim, name='FC2', kernel_initializer=self.xavier)

        # Merge the two branches
        Qout = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
        prediction = tf.argmax(Qout, axis=1)

        return prediction, Qout

    # noinspection PyAttributeOutsideInit
    def build_agent(self):

        self.rewards_episode = tf.placeholder(tf.float32, shape=[None, ], name='rewards_eps')
        tf.summary.histogram('rewards', self.rewards_episode, collections=['episode'])
        tf.summary.scalar('mean_reward', tf.reduce_mean(self.rewards_episode), collections=['episode'])
        # Initialize Model Inputs
        with tf.variable_scope('inputs'):
            self.state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='state')
            self.action = tf.placeholder(tf.int32, shape=[None, ], name='action')
            self.targetQ = tf.placeholder(tf.float32, shape=[None, ], name='targetQ')

        # Define Q-Network forward pass
        with tf.variable_scope('QNetwork'):
            self.predict, self.Qout = self.build_network(self.state)
            self.prob_predict = tf.multinomial(tf.nn.softmax(self.Qout), 1)

        tf.summary.histogram('actions', self.predict, collections=['train'])
        tf.summary.histogram('savings', tf.gather(self.act_grid, self.predict), collections=['train'])
        # Define Target Network forward pass
        with tf.variable_scope('TargetNetwork'):
            self.target_predict, self.target_Qout = self.build_network(self.state)

        # TODO: show this history
        with tf.name_scope("reward_estimation"):
            # These are filled with values coming from the queue
            self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")
            self.terminal = tf.placeholder(tf.float32, (None,), name="next_state_masks")
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

        with tf.name_scope('loss'):
            self.actions_ohe = tf.one_hot(self.action, self.action_dim, dtype=tf.float32)
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_ohe), axis=1)
            self.td_error = tf.squared_difference(self.targetQ, self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                # TODO: figure out what the bounds for the gradient ought to be
                # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="QNetwork")
                # optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # gvs = optimizer.compute_gradients(self.loss, train_vars)
                # capped_gvs = [(tf.clip_by_value(grad, -self.max_grad, self.max_grad), var) for grad, var in gvs]
                # self.train = optimizer.apply_gradients(capped_gvs)
            tf.summary.scalar('td', self.loss, collections=['train'])

        # Update target network with Q network
        with tf.name_scope("target_update"):
            self.target_update = []
            q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="QNetwork")
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="TargetNetwork")
            for v_source, v_target in zip(q_vars, target_vars):
                update_op = v_target.assign_sub(self.tau * (v_target - v_source))
                self.target_update.append(update_op)

            # Group all assignment operations together
            self.target_update = tf.group(*self.target_update)

        self.no_op = tf.no_op()
        self.train_summary_op = tf.summary.merge_all(key='train')
        self.episode_summary_op = tf.summary.merge_all(key='episode')

    def update_model(self, batch_data):

        sess = self.sess
        sess.run(self.train_phase)

        a_batch, r_batch, pre_s, post_s, t_batch = batch_data
        batch_size = len(r_batch)
        Q_actions = sess.run(self.predict, {self.state: post_s})
        Q_values = sess.run(self.target_Qout, {self.state: post_s})

        doubleQ = Q_values[range(batch_size), Q_actions]  # Get Q-values corresponding to actions
        targetQ = r_batch + (self.beta * doubleQ * -t_batch)  # TODO: determine negation procedure

        # Whether to produce summary
        if self.training_step % self.sum_freq == 0:
            op = self.train_summary_op
        else:
            op = self.no_op

        # Perform update op
        feed = {self.state: pre_s, self.targetQ: targetQ, self.action: a_batch}
        _, summary = sess.run([self.train, op], feed)

        sess.run(self.test_phase)
        sess.run(self.target_update)

        self.writer.add_summary(summary, global_step=self.training_step)
        self.training_step += 1
