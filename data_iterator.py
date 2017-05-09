import os
import sys

import random
import pickle
import itertools
import logging

import numpy as np

from SS_dataset import SSIterator

logger = logging.getLogger(__name__)

def create_padded_batch(state, data):
	mx = state['seqlen']
	n = state['bs']

	max_enc_len = 0
	max_dec_len = 0


	for i in range(state['bs']):
		if len(data[i][0]) > max_enc_len:
			max_enc_len = len(data[i][0])
		if len(data[i][1]) > max_dec_len:
			max_dec_len = len(data[i][1])

	X = np.zeros((n ,max_enc_len), dtype='int32')
	X_mask = np.zeros((n ,max_enc_len), dtype='float32')
	Y = np.zeros((n, max_dec_len), dtype='int32')
	Y_mask = np.zeros((n, max_dec_len), dtype='float32')

	for i in range(state['bs']):
		s_enc = np.array(data[i][0])
		s_enc_mask = np.ones_like(s_enc)
		z_enc = np.zeros(max_enc_len - len(s_enc))
		X[i] = np.concatenate((s_enc, z_enc))
		X_mask[i] = np.concatenate((s_enc_mask, z_enc))

		s_dec = np.array(data[i][1])
		s_dec_mask = np.ones_like(s_dec)
		z_dec = np.zeros(max_dec_len - len(s_dec))
		Y[i] = np.concatenate((s_dec, z_dec))
		Y_mask[i] = np.concatenate((s_dec_mask, z_dec))

	#return a dictionary
	return {'x':np.transpose(X),
			'x_mask':np.transpose(X_mask),
			'y':np.transpose(Y),
			'y_mask':np.transpose(Y_mask)
			}

class Iterator(SSIterator):
	def __init__(self, source_file, batch_size, **kwargs):
		SSIterator.__init__(self, source_file, batch_size,
							max_len=kwargs.pop('max_len', -1),
							use_infinite_loop=kwargs.pop('use_infinite_loop', False))

		self.k_batches = kwargs.pop('sort_k_batches', 20)
		
		self.state = kwargs.pop('state', None)

		self.batch_iter = None

	def get_homogenous_batch_iter(self, batch_size=-1):
		while True:
			batch_size = self.batch_size if batch_size== -1 else batch_size
			data = []
			for k in range(self.k_batches):
				batch = SSIterator.next(self)
				if batch:
					data.append(batch)

			if not len(data):
				return

			number_of_batches = len(data)
			data = list(itertools.chain.from_iterable(data))#each elem is a line of data

			#data_x = []
			#data_y = []
			#for i in range(len(data)):
			#	data_x.append(data[i])
				#data_y.append(data[i][1])
			data_x = data
			x = np.asarray(list(itertools.chain(data_x)))
			lens = np.asarray([map(len, x)])
			order = np.argsort(lens.max(axis=0))

			for k in range(number_of_batches):
				indices = order[k * batch_size:(k + 1) * batch_size]
				batch = create_padded_batch(self.state, x[indices])

				if batch:
					yield batch

	def start(self):
		SSIterator.start(self)
		self.batch_iter = None

	def next(self, batch_size=-1):
		if not self.batch_iter:
			self.batch_iter = self.get_homogenous_batch_iter(batch_size)
		try:
			batch = next(self.batch_iter)
		except StopIteration as e:
			raise e
		return batch

def get_batch_iterator(state):

	train_data = Iterator(
						state['train_data_file'],
						int(state['bs']),
						state=state,
						use_infinite_loop=True,
						max_len=state['seqlen'])
						
	'''valid_data = Iterator(
						state['valid_data_file'],
						int(state['bs']),
						state=state,
						max_len=state['seqlen'])
'''
	return train_data#, valid_data

if __name__ == "__main__":
	'''import ipdb
	ipdb.set_trace()
	state = {}
	state['train_data_file'] = '/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/data_net.pkl'
	state['bs'] = 16
	state['seqlen'] = -1
	train_data = get_batch_iterator(state)
	train_data.start()
	
	for i in range(3):
		batch = train_data.next()
		print i
		print batch['x']'''