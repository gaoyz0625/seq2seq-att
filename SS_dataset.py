import os
import gc

import logging
import threading
import Queue
import cPickle
import copy	
import numpy as np

logger = logging.getLogger(__name__)

class SSFetcher(threading.Thread):
	def __init__(self, parent):
		threading.Thread.__init__(self)
		self.parent = parent
		self.rng = np.random.RandomState(self.parent.seed)
		self.indexes = np.arange(parent.data_len)

	def run(self):
		diter = self.parent
		self.rng.shuffle(self.indexes)
		#import ipdb
		#ipdb.set_trace()
		offset = 0
		while not  diter.exit_flag:
			last_batch = False
			lines = []
			while len(lines) < diter.batch_size:
				if offset == diter.data_len:
					if not diter.use_infinite_loop:
						last_batch = True
						break
					else:
						self.rng.shuffle(self.indexes)
						offset = 0

				index = self.indexes[offset]
				s = diter.data[index]
				offset += 1

				if diter.max_len == -1 or len(s) <=diter.max_len:
					lines.append(s)
			if len(lines):
				diter.queue.put(lines)
			if last_batch:
				diter.queue.put(None)
				return

class SSIterator(object):
	def __init__(self,
				data_file,
				batch_size,
				seed=1234,
				max_len=-1,
				use_infinite_loop=True,
				dtype="int32"):
		self.data_file = data_file
		self.batch_size = batch_size
		self.seed = seed
		self.use_infinite_loop = use_infinite_loop
		args = locals()
		args.pop("self")
		self.__dict__.update(args)
		self.load_file()
		self.exit_flag = False

	def load_file(self):
		self.data = cPickle.load(open(self.data_file, 'r'))
		self.data_len = len(self.data)
		logger.debug('Data len is %d' % self.data_len)

	def start(self):
		self.exit_flag = False
		self.queue = Queue.Queue(maxsize=1000)
		self.gather = SSFetcher(self)
		self.gather.daemon = True
		self.gather.start()
	
	def __del__(self):
		if hasattr(self, 'gather'):
			self.gather.exitFlag = True
			self.gather.join()
			
	def __iter__(self):
		return self

	def next(self):
		if self.exit_flag:
			return None
		
		#alive = self.gather.isAlive()
		#print alive

		#print self.queue.qsize()
		batch = self.queue.get()
		if not batch:
			self.exit_flag =True
		return batch
