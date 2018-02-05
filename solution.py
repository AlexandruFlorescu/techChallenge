from __future__ import division # to run float division on TF 
from math import log
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import sys


def label_encoding(dt):
	lenc = preprocessing.LabelEncoder()
	for i in range(len(dt)):
		dt = lenc.fit_transform(dt)
	return dt

def read_data():
	with open('train.txt') as documents:
		train_dataset = documents.readlines()
		train_dataset_split = [doc.split() for doc in train_dataset]
		train_dataset = [doc[1:] for doc in train_dataset_split]
		train_labels = [doc[0] for doc in train_dataset_split]
	with open('test.txt') as documents:
		test_dataset = documents.readlines()
		test_dataset_split = [doc.split() for doc in test_dataset]
		test_dataset = [doc[1:] for doc in test_dataset_split]
		test_labels = [doc[0] for doc in test_dataset_split]
	return train_dataset, label_encoding(train_labels), test_dataset, label_encoding(test_labels)

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

def doc_to_vec(doc, idfs, vocab):
	feats = np.zeros(len(vocab))
	for word in doc:
		feats[vocab.index(word)] = tf(word, doc) * idfs[word]
	return feats

if __name__ == '__main__':
	train_dataset, train_labels, test_dataset, test_labels = read_data()
	print 'Training dataset has: {} documents'.format(len(train_dataset))
	print 'Testing dataset has: {} documents'.format(len(test_dataset))
	print 'Building vocabularies...',
	sys.stdout.flush()
	train_vocab = set([])
	test_vocab = set([])
	for doc in train_dataset:
		train_vocab = train_vocab.union(set(doc))
	for doc in test_dataset:
		test_vocab = test_vocab.union(set(doc))
	print 'Done'

	print 'Computing idfs on vocabularies...',
	sys.stdout.flush()
	train_idfs = {}
	test_idfs = {}
	for word in train_vocab:
		if word not in train_idfs:
			train_idfs[word] = idf(word, train_dataset)
	for word in test_vocab:
		if word not in test_idfs:
			test_idfs[word] = idf(word, test_dataset)
	print 'Done'

	print 'Converting documents to numerical features...',
	sys.stdout.flush()
	train_vectorized = [doc_to_vec(doc, train_idfs, list(train_vocab)) for doc in train_dataset]
	test_vectorized = [doc_to_vec(doc, train_idfs, list(test_vocab)) for doc in test_dataset]
	print 'Done'

	print 'Clustering using KMeans algorithm...',
	sys.stdout.flush()
	kmeans = KMeans(n_clusters = 7,random_state = 0).fit(train_vectorized)
	print 'Done'
	print 'KMeans(7): Accuracy on the training subset: {:.3f}'.format(metrics.adjusted_mutual_info_score(train_labels, kmeans.labels_))#kmeans.score(train_vectorized, train_labels))
	# sys.stdout.flush()
	# print 'KMeans(7): Accuracy on the test subset: {:.3f}'.format(metrics.adjusted_mutual_info_score(kmeans.fit(test_vectorized).score(test_vectorized, test_labels))

	print 'Clustering using Decision Trees algorithm...',
	sys.stdout.flush()
	tree = DecisionTreeClassifier(random_state = 0)
	tree.fit(train_vectorized, train_labels)
	print 'Done'
	print 'DTree: Accuracy on the training subset: {:.3f}'.format(tree.score(train_vectorized, train_labels))
	sys.stdout.flush()
	print 'DTree: Accuracy on the test subset: {:.3f}'.format(tree.score(test_vectorized, test_labels))


	# np.set_printoptions(threshold=np.nan)
	# print kmeans.labels_, train_labels
	# print normalized_mutual_info_score(kmeans.labels_, train_labels)

	# print len(train_dataset[0]), len(doc_to_vec(train_dataset[0], idfs))
