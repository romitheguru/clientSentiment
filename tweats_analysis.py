import nltk
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

# This will used hold the featured words appears in data
word_features = []

def get_word_features(train_data):
	all_words = []
	for token_list in train_data['tokens']:
		all_words.extend(token_list)
	wordlist = nltk.FreqDist(all_words)
	word_features.extend(wordlist.keys())


def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features


def get_train_data():
	train = pd.read_csv('feeds_train.tsv', sep='\t')
	tokenizer = RegexpTokenizer(r'\w+')
	train['tokens'] = train['Feed'].map(lambda x: tokenizer.tokenize(x))
	return train


def train_data_processing(train_data, feature_function):
	feeds = []
	for i in xrange(len(train_data)):
		tup = (train_data.iloc[i][2],str(train_data.iloc[i][0]))
		feeds.append(tup)
	training_set = nltk.classify.apply_features(feature_function, feeds)
	return training_set


def get_test_features(test_data, feature_function):
	tokenizer = RegexpTokenizer(r'\w+')
	test_data['tokens'] = test_data[0].map(lambda x: tokenizer.tokenize(x))
	feeds = []
	for i in xrange(len(test_data)):
		row = map(str, test_data.iloc[i][1])
		tup = (row)
		feeds.append(tup)
	return feeds


def train_model(training_set):
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	f = open('tweat_feed_classifier.pickle', 'wb')
	pickle.dump(classifier, f)
	f.close()


def predict_model(test_set):
	f = open('tweat_feed_classifier.pickle', 'rb')
	model = pickle.load(f)
	predictor = []
	for data in test_set:
		predictor.append(model.prob_classify(extract_features(data)))
	predictions = []
	for dist in predictor:
		temp = []
		for label in dist.samples():
			temp.append(dist.prob(label))
		predictions.append(temp)
	return predictions

if __name__ == '__main__':
	test = pd.read_excel('TweeterFeeds.xlsx', header=None)
	train = get_train_data()
	training_set = train_data_processing(train, extract_features)
	test_set = get_test_features(test, extract_features)
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	tweet = 'Larry is my friend'
	dist = classifier.prob_classify(extract_features(tweet.split()))
	for label in dist.samples():
		print("%s: %f" % (label, dist.prob(label)))

	train_model(training_set)
	predictions = predict_model(test_set)
	print predictions
