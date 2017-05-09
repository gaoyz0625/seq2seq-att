import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_rnn, static_bidirectional_rnn


logger = logging.getLogger(__name__)


class RNNEncoderDecoder(object):

	def __init__(self, state):
		self.state = state

		self.embedding_enc = tf.Variable(self.init_matrix([state['vocab_size_enc'], state['emb_dim']]))
		self.embedding_dec = tf.Variable(self.init_matrix([state['vocab_size_dec'], state['emb_dim']]))
		self.params = []
		#self.params.append(self.embedding)

		self.add_placeholder()
		#self.build_network()

	def add_placeholder(self):
		self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.state['bs']]) # sen_len x bs
		self.X_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.state['bs']])
		self.Y = tf.placeholder(dtype=tf.int32, shape=[None, self.state['bs']]) # sen_len x bs
		self.Y_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.state['bs']])

	def build_network(self):
		#init_state = self.mlp_layer(hidden_state)
		init_state, context = self.encode()
		self.cost = self.attention_decode(init_state, context)
		#self.cost = self.decode(init_state, context)


	'''def mlp_layer(self, input_emb, input_mask):
		with tf.variable_scope('ff_layer'):
			self.W = tf.Variable(self.init_matrix([state['emb_dim'], state['hidden_dim']]))
			self.b = tf.Variable(self.init_matrix(state['hidden_dim']))
			self.params.extend([self.W, self.b])
			return tf.nn.tanh(input_emb * self.W + self.b)


	def RNN_layer(self, input_emb, sequence_length, init_state=None, input_mask=None, is_bidirectional=False,is_multilayer=False):
		with tf.variable_scope('gru_layer') as scope:
			gru_cell = GRUCell(self.state['hidden_dim'], activation=tf.relu)
			init_state = gru_cell.zero_state(state['bs'])
			rnn_outputs, hidden_state = static_rnn(gru_cell, input_emb, initial_state=init_state, sequence_length=sequence_length,scope=scope)
			#hidden_state = self.rnn_template(input_emb, gru_cell, return_seq=True, init_state=init_state, scope=scope)
			#add mask
			
			return rnn_output, hidden_state
			
	def multiRNNlayer(self, input_emb, sequence_length, init_state_f=None, init_state_b=None, input_mask=None):
		states_f = []
  		states_b = []
  		init_state_f = []
  		init_state_b = []
  		prev_layer = input_emb
		with tf.variable_scope('multi_rnn'):
			forward_gru_cell = [GRUCell(self.state['hidden_dim'], activation=tf.relu)] * state['level']
			backward_gru_cell = [GRUCell(self.state['hidden_dim'], activation=tf.relu)] * state['level']

			assert len(forward_gru_cell) ==	state['level']
			#init_state = gru_cell.zero_state(state['bs'])
			
			for level, (f_cell, b_cell) in enumerate(zip(forward_gru_cell, backward_gru_cell)):
				if init_state_f:
					init_state_f = init_state_f[level]
				if init_state_b:
					init_state_b = init_state_b[level]

				with tf.variable_scope("cell_%d"%level) as cell_scope:
					prev_layer, state_f, state_b = static_bidirectional_rnn(f_cell,
																			b_cell,
																			prev_layer,
																			initial_state_fw=init_state_f,
																			initial_state_bw=init_state_b,
																			sequence_length=sequence_length,
																			scope=cell_scope)
				states_f.append(state_f)
				states_b.append(state_b)
			return prev_layer, tuple(states_f), tuple(states_b)

	def maxout_layer(self):
		with tf.variable_scope('maxout_layer'):
			pass
	'''
	def GRU_layer(self, emb, mask=None, init_state=None):

		n_steps = tf.shape(emb)[0]
		tf.assert_rank(emb, 3)
		n_samples = tf.shape(emb)[1]
		if mask == None:
			mask = tf.fill([n_samples, None], tf.constant(1.0))
		if init_state == None:
			init_state = tf.fill([self.state['bs'], self.state['hidden_dim']], 0.0)

		with tf.variable_scope('gru') as scope:
			Ur = tf.Variable(self.init_matrix([self.state['emb_dim'], self.state['hidden_dim']]))
			Wr = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			Uu = tf.Variable(self.init_matrix([self.state['emb_dim'], self.state['hidden_dim']]))
			Wu = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			U = tf.Variable(self.init_matrix([self.state['emb_dim'], self.state['hidden_dim']]))
			W = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))

		var = [Ur, Wr, Uu, Wu, U, W]
		hidden_state = tf.TensorArray(dtype=tf.float32, size=n_steps,
										dynamic_size=True, clear_after_read=False, infer_shape=True)
		hidden_state = hidden_state.write(0, init_state)
		#unstack emb and mask
		
		#emb = tf.unstack(emb)
		#mask = tf.unstack(mask)
		#i = 0 
		def _step(i, emb, mask, hidden_state, var):
			import ipdb
			ipdb.set_trace()
			'''if tf.identity(i, tf.constant(0, dtype=tf.int32)) is not None:
				h_ = hidden_state.read(i)
			else:
				h_ = hidden_state.read(i-1)
			'''
			a = hidden_state.read(i-1)
			b = hidden_state.read(i)
			cond = True
			h_ = tf.where(cond, b, a)
			#h_ = hidden_state.read(i)			
			emb_i = emb[i]
			mask_i = mask[i]

			reset = tf.nn.sigmoid(tf.matmul(emb_i, var[0]) + tf.matmul(h_, var[1]))
			update = tf.nn.sigmoid(tf.matmul(emb_i, var[2]) + tf.matmul(h_, var[3]))

			h_hat = tf.nn.tanh(tf.matmul(emb_i, var[4]) + tf.matmul(tf.multiply(reset, h_), var[5]))
			h = update * h_ + (1 - update) * h_hat

			#masking
			h = mask_i[:, None] * h + (1 - mask_i)[:, None] * h_
			hidden_state = hidden_state.write(i, h)

			return i+1, emb, mask, hidden_state, var

		_, _, _, hidden_state, _= tf.while_loop(cond=lambda i, _1, _2, _3, _4: i < n_steps,
						body=_step,
						loop_vars=(tf.constant(0, dtype=tf.int32), emb, mask, hidden_state, var))

		return hidden_state.stack()

	def Cond_GRU_layer(self, emb, mask=None, context=None, context_mask=None, init_state=None, one_step=False):
		if one_step:
			n_steps = 1
		else:
			n_steps = tf.shape(emb)[0]
		tf.assert_rank(emb, 3)
		n_samples = tf.shape(emb)[1]
		if init_state == None:
			init_state = tf.fill([self.state['bs'], state['hidden_dim']], 0)

		with tf.variable_scope('cond_gru'):
			Ur = tf.Variable(self.init_matrix([self.state['emb_dim'], self.state['hidden_dim']]))
			Wr = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			Uu = tf.Variable(self.init_matrix([self.state['emb_dim'], self.state['hidden_dim']]))
			Wu = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			U = tf.Variable(self.init_matrix([self.state['emb_dim'], self.state['hidden_dim']]))
			W = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			Wc = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			vc = tf.Variable(self.init_matrix([self.state['hidden_dim']]))
			Wh = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			Uh = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['hidden_dim']]))
			bh = tf.Variable(self.init_matrix([self.state['hidden_dim']]))

		var = [Ur, Wr, Uu, Wu, U, W, Wc, vc, Wh, Uh, bh]

		hidden_state = tf.TensorArray(dtype=tf.float32, size=n_steps+1,
										dynamic_size=True, clear_after_read=False, infer_shape=True) 
		hidden_state = hidden_state.write(0, init_state)

		assert len(var) == 11
		
		def _step(i, emb, mask, hidden_state, ctx, ctx_mask, var):
			'''if tf.identity(i, tf.constant(0, dtype=tf.int32)) is not None:
				h_ = hidden_state.read(i)
			else:
				h_ = hidden_state.read(i-1)
				'''
			h_ = hidden_state.read(i)
			emb_i = emb[i]
			mask_i = mask[i]
			#attention
			#import ipdb
			#ipdb.set_trace()

			reset = tf.nn.sigmoid(tf.matmul(emb_i, var[0]) + tf.matmul(h_, var[1]))
			update = tf.nn.sigmoid(tf.matmul(emb_i, var[2]) + tf.matmul(h_, var[3]))
			h_hat = tf.nn.tanh(tf.matmul(emb_i, var[4]) + tf.matmul(tf.multiply(reset, h_), var[5]))
			h1 = update * h_ + (1 - update) * h_hat
			h1 = mask_i[:, None] * h1 + (1 - mask_i)[:, None] * h_
			#calculate alpha
			pstate = tf.matmul(h1, var[6])
			pctx = ctx + pstate[None, :, :]
			pctx = tf.nn.tanh(pctx)
			#alpha = tf.tensordot(pctx, var[7], [[2], [0]])
			alpha = pctx * var[7][None, None, :] #+ bc		
			alpha = tf.reduce_sum(alpha, axis=2)
			alpha = tf.exp(alpha)
			
			if ctx_mask is not None:
				alpha = alpha * ctx_mask
			alpha = alpha / tf.reduce_sum(alpha, axis=0)

			ctx_ = tf.reduce_sum(alpha[:, :, None] * ctx, axis=0)
			pstate2 = tf.matmul(h1, var[8])
			pctx2 = tf.matmul(ctx_, var[9])
			h2 = tf.nn.tanh(pstate2 + pctx2 + var[10])
			h2 = mask_i[:, None] * h2 + (1 - mask_i)[:, None] * h1

			hidden_state = hidden_state.write(i, h2)
			return i+1, emb, mask, hidden_state, ctx , ctx_mask, var

		_, _, _, hidden_state, _, _, _ = tf.while_loop(cond=lambda i, _1, _2, _3, _4, _5, _6: i< n_steps+1,
						body=_step,
						loop_vars=(tf.constant(0, dtype=tf.int32), emb, mask, hidden_state, context, context_mask, var))
		
		return hidden_state.stack()

	def encode(self):
		with tf.name_scope('encoder'):
			emb = tf.nn.embedding_lookup(self.embedding_enc, self.X)
			mask = self.X_mask
			hidden_state = self.GRU_layer(emb, mask)
			context = hidden_state
			#context = tf.unstack(hidden_state)
			last_hidden_state = context[-1]
			return last_hidden_state, context

	'''def attention_encode(self):
		with tf.name_scope('att_encoder'):
			input_emb = tf.nn.embedding_lookup(self.embedding, self.X)
			rnn_outputs, forward_hidden_state, back_hidden_state = self.multiRNNlayer(input_emb, sequence_length, input_mask=self.X_mask)
			return rnn_outputs, forward_hidden_state, back_hidden_state
'''
	'''def encode1(self):
		with tf.name_scope('encoder'):
			input_emb = tf.nn.embedding_lookup(self.embedding, self.X)
			mlp_output = self.mlp_layer(input_emb,self.X_mask)
			sequence_length = 
			rnn_outputs, hidden_state = self.RNN_layer(mlp_output, sequence_length,input_mask=self.X_mask)
			return rnn_outputs, hidden_state
'''
	'''def decode(self, init_state, context):
		with tf.name_scope('decoder'):
			input_emb = tf.nn.embedding_lookup(self.embedding, self.Y)
			Y_shape = tf.getshape(self.Y).as_list()
			rnn_outputs, hidden_state = self.RNN_layer(input_emb, self.seq_len, init_state=encode_state, input_mask= self.Y_mask)
			rnn_output = tf.stack(rnn_outputs, axis=0)
			with tf.variable_scope('output_layer'):
				self.W = tf.get_variable('W', self.init_matrix([state['hidden_dim'], state['']]))
				self.b = tf.get_variable('b', self.init_matrix([state['']]))
			with tf.variable_scope('output_layer', reuse=True):
				W = tf.get_variable('w')
				b = tf.get_variable('b')
				logit = tf.matmul(rnn_output, w) + b
				probs = -tf.log(tf.nn.softmax(logit))
				#shape of probs: timesteps x batch_size x n_vocab
				y_flat = tf.reshape(self.Y, [None])
				y_flat_idx = 
				cost = tf.reshape(probs, [None])[y_flat_idx]
				cost = tf.reshape(cost, [Y_shape[1], Y_shape[0]])
				cost = tf.reduce_sum(cost * Y_mask, axis=0)
			return cost
'''
	def attention_decode(self, init_state, context):
		with tf.name_scope('decoder'):
			
			emb = tf.nn.embedding_lookup(self.embedding_dec, self.Y)
			Y_shape = tf.shape(self.Y)#.as_list()
			mask = self.Y_mask
			context_mask = self.X_mask
			hidden_state = self.Cond_GRU_layer(emb, mask=mask, context=context, context_mask=context_mask, init_state=init_state,one_step=False)
			#calculate probability

			with tf.variable_scope('output_layer'):
				W = tf.Variable(self.init_matrix([self.state['hidden_dim'], self.state['vocab_size_dec']]))
				b = tf.Variable(self.init_matrix([self.state['vocab_size_dec']]))
				#logit = tf.matmul(hidden_state, W[None, :, :]) + b
				#logit = hidden_state * W[None, :, :] + b
				#logit = tf.multiply(hidden_state, W[None, :, :])
				logit = tf.tensordot(hidden_state, W, [[2], [0]])
				probs = -tf.log(tf.nn.softmax(logit))
				#shape of probs: timesteps x batch_size x n_vocab
				y_flat = tf.reshape(self.Y, [-1])
				y_flat_idx = self.state['vocab_size_dec'] * tf.range(start=0, limit=tf.shape(y_flat)[0]) + y_flat
				cost = tf.nn.embedding_lookup(tf.reshape(probs, [-1]), y_flat_idx)
				#cost = tf.reshape(probs, [-1])[y_flat_idx]
				#cost = tf.reshape(probs, [-1])

				cost = tf.reshape(cost, [Y_shape[0], Y_shape[1]])
				cost = tf.reduce_sum(cost * mask, axis=0)
				cost = tf.reduce_sum(cost)
			return cost
	def init_operation(self, sess):

		self.train_var = tf.trainable_variables()
		init_op = tf.variables_initializer(self.train_var)
		#init_op = tf.local_variables_initializer()
		sess.run(init_op)
		#return init_op
	def train_operation(self, DataIterator, sess):
		import ipdb
		ipdb.set_trace()

		self.optimizer = tf.train.GradientDescentOptimizer(self.state['lr'])
		self.train_op = self.optimizer.minimize(self.cost)
		#train_var = tf.trainable_variables()
		'''gradient = self.optimizer.compute_gradients(self.cost, self.train_var)
		gradient = tf.clip_by_global_norm((gradient,self.train_var), self.state['max_grad_norm'])
		self.train_op = self.optimizer.apply_gradients(zip(gradient, self.train_var))
		'''
		#return train_op
		step = 0
		train_cost = 0
		
		DataIterator.start()

		while step < self.state['loop_iters']:
			
			batch = DataIterator.next()

			if not batch:
				logger.debug("Got None...")
				break

			X_data = batch['x']
			X_mask = batch['x_mask']
			Y_data = batch['y']
			Y_mask = batch['y_mask']

			feed = {self.X: X_data,
					self.X_mask: X_mask,
					self.Y: Y_data,
					self.Y_mask: Y_mask
					}
			_, cost = sess.run([self.train_op, self.cost], feed_dict=feed)
			#_, cost = sess.run([self.train_op, self.cost], feed_dict=feed)
			train_cost += cost
			print train_cost
			step += 1


	def generate_sample(self):

		pass
	def beam_search():
		pass

	def ortho_weight(self, ndim):
		W = np.random.randn(ndim, ndim)
		u, s, v = np.linalg.svd(W)
		return u.astype('float32')
	
	def norm_weight(self, nin, nout, scale=0.1, ortho=True):
		if nout is None:
			nout = nin
		if nout == nin and ortho:
			W = ortho_weight(nin)
		else:
			W = scale * np.random.randn(nin, nout)
		return W.astype('float32')

	def init_matrix(self, shape):
		return tf.random_normal(shape, stddev = 0.1)

