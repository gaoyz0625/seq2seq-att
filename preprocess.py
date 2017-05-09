import os
import sys
import io
import nltk
import re
import cPickle
import operator
import numpy as np
from contextlib import nested
UTTERANCE_PATH = "/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/movie_dialog/movie_lines.txt"
CONVERSATION_PATH = "/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/movie_dialog/movie_conversations.txt"

START_TOKEN = 'SOS'
END_TOKEN = 'EOS'
UNK_TOKEN = 'UNK'
MASK_TOKEN = 'MASK'

vocab_size = 50000

def preprocess():

	#import ipdb
	#ipdb.set_trace()
	
	utterance = {}
	with open(UTTERANCE_PATH, 'r') as utter_file:
		while True:

			line = utter_file.readline()
			if not line:
				break
			line = line.split("+++$+++")
			utterance[re.sub(r'\s', '', line[0])] = re.sub(r'\n', '', re.sub(r'\s', '', line[-1], count=1))

		#utter_file.close()

	with open(CONVERSATION_PATH, 'r') as dialog_file:
		f = open("data.txt", 'w')
		count = 0
		while True:
			line = dialog_file.readline()
			if not line:
				break
			line = line.split("+++$+++")
			content = line[-1]
			content = re.sub(r'\s', '', content)
			r = re.compile(r'[\[\'\]]')
			content = r.sub('', content)
			content = content.split(',')
			length = len(content)
			for i in range(length // 2):
				#print utterance[content[0]]
				#print utterance[content[1]]
				utterance2 = '%s %s %s' % (START_TOKEN, utterance[content[2*i+1]], END_TOKEN)
				file_line = '+++$+++'.join([utterance[content[2*i]], utterance2])
				#print file_line
				f.write("%s\n" % file_line)
				count += 1
		#dialog_file.close()
		#f.close()
		return count

def statistics(filename):

	#word2idx = {}
	#idx2word = {}
	utterances = []
	import ipdb
	ipdb.set_trace()
	with open(filename, 'r') as f:
		i = 0
		while True:
			line = f.readline()
			i += 1
			if not line or i > 10000:
				break
			#line = line[:-1]
			line = line.split('+++$+++')
			for u in line:
				utterances.append(u)
		#f.close()

	utter_len = len(utterances)
	print utterances[7405]
	#tokenized_utterances = [nltk.word_tokenize(u) for u in utterances]
	tokenized_utterances = []
	for i, u in enumerate(utterances):
		tokenized_utterances.append(nltk.word_tokenize(u))
		if (i+1) % 100 == 0:
			print i
	#tokenized_utterances = []
	'''for i, u in enumerate(utterances):
		print i
		tokenized_utterance = nltk.word_tokenize(u)
	'''
	word_freq = nltk.FreqDist(nltk.chain(*tokenized_utterances))
	item = word_freq.items()
	vocab = sorted(item, key=lambda x:(x[1], x[0]), reverse=True)[:vocab_size]
	sorted_vocab = sorted(vocab, key=operator.itemgetter(1))

	idx2word = [MASK_TOKEN, UNK_TOKEN] + [x[0] for x in sorted_vocab]
	word2idx = dict([w, i] for i ,w in enumerate(idx2word))

	#replace the lowly frequent word with 'unk'
	for i, sent in enumerate(tokenized_utterances):
		tokenized_utterances[i] = [w if w in idx2word else UNK_TOKEN for w in sent]

	idx2word_file = open('idx2word.pkl', 'w')
	word2idx_file = open('word2idx.pkl', 'w')
	cPickle.dump(idx2word, idx2word_file)
	cPickle.dump(word2idx, word2idx_file)
	#idx2word_file.close()
	#word2idx.close()
	#return word2idx, idx2word
train_enc = '/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/dataset/train.enc.txt'
train_dec = '/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/dataset/train.dec.txt'
vocab_enc_file = '/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/dataset/vocab20000.enc.txt'
vocab_dec_file = '/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/dataset/vocab20000.dec.txt'

def replace():

	#utterances_enc = []
	#utterances_dec = []
	#import ipdb
	#ipdb.set_trace()
	vocab_enc = []
	vocab_dec = []
	
	with io.open(vocab_enc_file, 'r') as f1, open(vocab_dec_file, 'r') as f2:
		while True:
			line = f1.readline()
			if not line:
				break
			line = line.rstrip()
			vocab_enc.append(line)
		vocab_enc[0] = u'_PAD'
		word2idx_enc = dict([w, i] for i, w in enumerate(vocab_enc))

		while True:
			line_ = f2.readline()
			if not line_:
				break
			line_ = line_.rstrip()
			vocab_dec.append(line_)
		vocab_dec[0] = u'_PAD'
		word2idx_dec = dict([w, i] for i, w in enumerate(vocab_dec))

		print 'num of vocab_enc', len(vocab_enc)
		print 'num of vocab-dec', len(vocab_dec)

		#import ipdb
		#ipdb.set_trace()
		with open(train_enc, 'r') as f1, open(train_dec, 'r') as f2, open('data_net1.pkl', 'w') as data:
			utterances = []
			i = 0
			while True:
				'''i += 1
				if i % 3000 == 0:
					print i'''
				line = f1.readline()
				line_ = f2.readline()
				if not line or not line_:
					break
				line = nltk.word_tokenize(line)
				line_ = nltk.word_tokenize(line_)
				
				sentence_enc = []
				keys_enc = word2idx_enc.keys()
				#a = [w for w in line]
				#print a
				for w in line:
					if w in keys_enc:
						sentence_enc.append(word2idx_enc[w])
					else:
						sentence_enc.append(3)
				sentence_enc = [1] + sentence_enc + [2]
				
				sentence_dec = []
				keys_dec = word2idx_dec.keys()
				for w in line_:
					if w in keys_dec:
						sentence_dec.append(word2idx_dec[w])
					else:
						sentence_dec.append(3)
				sentence_dec = [1] + sentence_enc + [2]

				utterances.append([sentence_enc, sentence_dec])

			print len(utterances)
			utterances = np.array(utterances)
			cPickle.dump(utterances, data)
			#utterances_enc.append([w for w in line])
			#utterances_dec.append([w for w in line_])
			return

if __name__ == "__main__":
	#import sys
	#reload(sys)
	#sys.setdefaultencoding('gb18030')
	#c = preprocess()
	#statistics('data.txt')
	#print len(idx2word)
	replace()
	with open('data_net1.pkl', 'r') as data:
		utterances = cPickle.load(data)

	print utterances[:5]