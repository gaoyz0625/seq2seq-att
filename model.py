import logging
import os

import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)

class Training_model(object):
	def __init__(self):
		self.train_info()

	def train_model(self, sess, state, DataIterator, model):
		
		step = 0
		train_cost = 0
		
		DataIterator.start()
		import ipdb
		ipdb.set_trace()

		while step < state['loop_iters']:
			
			batch = DataIterator.next()

			if not batch:
				logger.debug("Got None...")
				break

			X_data = batch['x']
			X_mask = batch['x_mask']
			Y_data = batch['y']
			Y_mask = batch['y_mask']

			feed = {model.X: X_data,
					model.X_mask: X_mask,
					model.Y: Y_data,
					model.Y_mask: Y_mask
					}
			
			_, cost = sess.run([model.train_op, model.cost], feed_dict=feed)
			train_cost += cost
			print train_cost
			step += 1
			#validation
			
		#logger.debug("Training end")
		
	def train_info(self):
		pass
	def save(self, sess):
		SAVE_PATH = os.path.join(state['workdir'], "my_model_params")
		self.saver = tf.train.Saver()
		self.saver.save(sess, SAVE_PATH, global_step = step)