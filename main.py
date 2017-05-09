import os
import sys

import logging 

import tensorflow as tf
import numpy as np

from state import *
from enc_dec import *
from model import *
from data_iterator import *

logger = logging.getLogger(__name__)

def main(_):
	state = prototype_state()

	logging.basicConfig(level=getattr(logging, state['debug_level']),\
						format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
	logger.debug("")

	
	#build model
	with tf.Session() as sess:

		enc_dec = RNNEncoderDecoder(state)
		enc_dec.build_network()
		enc_dec.init_operation(sess)
		#sess.run(init_op)
		#load data iterator
		DataIterator = get_batch_iterator(state)
		#DataIterator.start()
		#training
		enc_dec.train_operation(DataIterator, sess)
		
		#train = Training_model()
		#train.train_model(sess, state, DataIterator, enc_dec)
		#train.save(sess)

if __name__ == "__main__":
	tf.app.run()
