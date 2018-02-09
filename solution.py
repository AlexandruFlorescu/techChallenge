from __future__ import division # to run float division on TF 
from math import log
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import mixture
import numpy as np
import sys


def label_encoding(dt):
	''' encoding string categorical into numerical categorical '''
	lenc = preprocessing.LabelEncoder()
	for i in range(len(dt)):
		dt = lenc.fit_transform(dt)
	return dt

def read_data():
	''' reads input train and test from documents '''
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
	''' computes tf on the term and the document '''
	count = 0
	for word in doc:
		if word == term:
			count += 1
	return count / len(doc)

def idf(term, docs):
	''' computes idf on the term and the documents given. Laplace smoothing can be added '''
	nr_docs = len(docs) #+ 1 #laplace smoothing 
	nr_docs_with_term = 0 #+ 1 #laplace smoothing 
	for doc in docs:
		if term in doc:
			nr_docs_with_term +=1 
	return log(nr_docs/nr_docs_with_term)

def doc_to_vec(doc, idfs, vocab):
	''' converts given document to features vector '''
	feats = np.zeros(len(vocab))
	for word in doc:
		if word in vocab: # filter words that only appear in test_dataset
			feats[vocab.index(word)] = tf(word, doc) * idfs[word]
	return feats

def split_by_class(dataset, labels):
	''' Unused! split dataset by classes. '''
	classes_dataset = [[], [], [], []]
	for index, row in enumerate(dataset):
		classes_dataset[labels[index]].append(row)
	return classes_dataset


if __name__ == '__main__':

	train_dataset, train_labels, test_dataset, test_labels = read_data()
	print 'Training dataset has: {} documents'.format(len(train_dataset))
	print 'Testing dataset has: {} documents'.format(len(test_dataset))
	
	print 'Building vocabularies...',
	sys.stdout.flush()
	train_vocab = set([])
	# test_vocab = set([])
	for doc in train_dataset:
		train_vocab = train_vocab.union(set(doc))
	# for doc in test_dataset:
	# 	test_vocab = test_vocab.union(set(doc))
	print 'Done'

	print 'Computing idfs on vocabularies...',
	sys.stdout.flush()
	train_idfs = {}
	# test_idfs = {}
	for word in train_vocab:
		if word not in train_idfs:
			train_idfs[word] = idf(word, train_dataset)
	# for word in test_vocab:
	# 	if word not in test_idfs:
	# 		test_idfs[word] = idf(word, test_dataset)
	print 'Done'

	print 'Converting documents to numerical features...',
	sys.stdout.flush()
	train_vectorized = [doc_to_vec(doc, train_idfs, list(train_vocab)) for doc in train_dataset]
	test_vectorized = [doc_to_vec(doc, train_idfs, list(train_vocab)) for doc in test_dataset]
	print 'Done'

	print 'Scaling the data using MinMaxScaler...',
	sys.stdout.flush()
	scaler = MinMaxScaler()
	train_vectorized = scaler.fit_transform(train_vectorized)
	test_vectorized = scaler.fit_transform(test_vectorized)
	print 'Done'

	# turns out the same improvement on Kmeans can be obtained using normalization applied below

	# print 'Applying Principal Component Analysis...',
	# sys.stdout.flush()
	# pca = decomposition.PCA(n_components = 3)
	# train_vectorized = pca.fit_transform(train_vectorized)
	# test_vectorized = pca.fit_transform(test_vectorized)
	# print 'Done'


	# print 'Trying the GaussianMixture trick that Tibi suggested',
	# sys.stdout.flush()
	# gmm = mixture.BayesianGaussianMixture(covariance_type = 'spherical').fit(train_vectorized, train_labels)
	# # gmms = []
	# # data_by_class = split_by_class(train_vectorized, train_labels)
	# # for index,cs in enumerate(data_by_class[:1]):
	# # 	gmms.append(mixture.BayesianGaussianMixture( covariance_type='spherical'))
	# # 	gmms[index].fit(cs)
	# for i in range(len(test_labels)):
	# 	print test_labels[i], gmm.predict_proba(test_vectorized[i].reshape(1,-1))
	# print 'Done'

	print 'Classifying using logistic regression...',
	sys.stdout.flush()
	logreg = LogisticRegression(C=1e5).fit(train_vectorized, train_labels)
	print 'Done'
	print 'Accuracy on the training subset: {:.3f}'.format(logreg.score(train_vectorized, train_labels))
	sys.stdout.flush()
	print 'Accuracy on the test subset: {:.3f}'.format(logreg.score(test_vectorized, test_labels))


	print 'Clustering using Decision Trees algorithm...',
	sys.stdout.flush()
	tree = DecisionTreeClassifier(random_state = 0)
	tree.fit(train_vectorized, train_labels)
	print 'Done'
	print 'DTree: Accuracy on the training subset: {:.3f}'.format(tree.score(train_vectorized, train_labels))
	sys.stdout.flush()
	print 'DTree: Accuracy on the test subset: {:.3f}'.format(tree.score(test_vectorized, test_labels))

	print 'Clustering using KMeans algorithm...',
	sys.stdout.flush()
	train_vectorized = preprocessing.normalize(train_vectorized)

	kmeans = KMeans(n_clusters = 4,random_state = 0).fit(train_vectorized)
	print 'Done'
	# np.set_printoptions(threshold=np.nan) this allows to see all the values in the labels array
	print 'KMeans(4): Accuracy on the training subset: {:.3f}'.format(metrics.adjusted_mutual_info_score(train_labels, kmeans.labels_))#kmeans.score(train_vectorized, train_labels))

		
