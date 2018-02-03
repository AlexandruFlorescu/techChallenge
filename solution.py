from __future__ import division # to run float division on TF 
from math import log
from sklearn.cluster import KMeans
import numpy as np
import sys

def read_data():
	with open('train.txt') as documents:
		train_dataset = documents.readlines()
		train_dataset = [doc.split() for doc in train_dataset]
	with open('test.txt') as documents:
		test_dataset = documents.readlines()
		test_dataset = [doc.split() for doc in test_dataset]
	return train_dataset, test_dataset

def tf(term, doc):
	count = 0
	for word in doc:
		if word == term:
			count += 1
	return count / len(doc)

def idf(term, docs):
	nr_docs = len(docs) #+ 1 #laplace smoothing 
	nr_docs_with_term = 0 #+ 1 #laplace smoothing 
	for doc in docs:
		if term in doc:
			nr_docs_with_term +=1 
	return log(nr_docs/nr_docs_with_term)

def doc_to_vec(doc, idfs, corpus):
	feats = np.zeros(len(corpus))
	for word in doc:
		feats[corpus.index(word)] = tf(word, doc) * idfs[word]
	return feats

if __name__ == '__main__':
	train_dataset, test_dataset = read_data()
	print 'Training dataset has: {} documents'.format(len(train_dataset))
	print 'Testing dataset has: {} documents'.format(len(test_dataset))

	print 'Building corpus...',
	sys.stdout.flush()
	corpus = set([])
	for doc in train_dataset:
		corpus = corpus.union(set(doc))
	print 'Done'

	print 'Computing idfs on corpus...',
	sys.stdout.flush()
	idfs = {}
	for word in corpus:
		if word not in idfs:
			idfs[word] = idf(word, train_dataset)
	print 'Done'

	print 'Converting documents to vectorial format...',
	train_vectorized = [doc_to_vec(doc, idfs, list(corpus)) for doc in train_dataset]
	print 'Done'

	print 'Clustering using KMeans algorithm',
	sys.stdout.flush()
	kmeans = KMeans(n_clusters = 7).fit(train_vectorized)
	print 'Done'
	np.set_printoptions(threshold=np.nan)
	print kmeans.labels_

	# print len(train_dataset[0]), len(doc_to_vec(train_dataset[0], idfs))
