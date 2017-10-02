import os
import json
from csv import DictReader, DictWriter
import nltk

import numpy as np
from numpy import array
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score

SEED = 5


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features

# TODO: Add custom feature transformers for the movie review data

class pos_words(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pos_words = ["absolute", "amazing", "approve", "attractive", "awesome", "beautiful", "brilliant", "creative", "delight", "enchanting", "excellent", "exciting", "fabulous", "fantastic", "favorable", "fun", "friendly", "funny", "genius", "gorgeous", "good", "great", "happy", "imagine", "impress", "joy", "laugh", "love", "marvelous", "master", "masterpiece", "nice", "perfect", "pleasant", "popular", "positive", "remarkable", "respect", "reward", "right", "satisfactory", "simple", "smile", "success", "super", "terrific", "thrill", "victorious", "victory", "whole", "wonderful", "wondrous", "wow", "yes"]
        # self.pos_words = ["absolute", "amazing", "approve", "attractive", "awesome", "beautiful", "brilliant", "creative", "delight", "enchanting", "excellent", "exciting", "fabulous", "fantastic", "favorable", "fun", "friendly", "funny", "genius", "gorgeous", "good", "great"]

    def fit(self, examples):
        return self

    def transform(self, examples):
        # If the word contains one of the common negative words, increase the count
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            words = nltk.word_tokenize(ex)
            for word in words:
                for pw in self.pos_words:
                    if pw in word:
                        features[i, 0] += 1
            features[i, 0] /= len(words)
            i += 1

        return features


class neg_words(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.neg_words = ["appalling", "atrocious", "awful", "bad", "boring", "banal", "beware", "confused", "depressed", "deprived", "disgusting", "disrupt", "dismal", "dreadful", "dreary", "dark", "fail", "hate", "hideous", "horrible", "insane", "loser", "messy", "negative", "no", "offensive", "pain", "painful", "pest", "reject", "repulsive", "resent", "sad", "sorrow", "sick", "sorry", "stupid", "terrible", "tired", "ugly", "unhappy", "upset", "zero"]
        # self.neg_words = ["appalling", "atrocious", "awful", "bad", "boring", "banal", "beware", "confused", "depressed", "deprived", "disgusting", "disrupt", "dismal", "dreadful"]
        # self.neg_words = ["appalling", "atrocious", "awful"]

    def fit(self, examples):
        return self

    def transform(self, examples):
        # If the word contains one of the common positive words, increase the count
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            words = nltk.word_tokenize(ex)
            for word in words:
                for pw in self.neg_words:
                    if pw in word:
                        features[i, 0] += 1
            features[i, 0] /= len(words)
            i += 1

        return features


class Suffixes(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.suffixes = ['e', 'the', 's', 'of', 'd', 'he', 'a', 't', 'n', 'to', 'in', 'and', 'y', 'r', 'is', 'f', 'o', 'ed', 'on','nd', 'as', 'l', 'g', 'at', 'ng', 'er', 'it', 'ing', 'h', 'or', 'es', 're', 'i', 'an', 'was', 'be', 'his', 'for', 'm', 'ly', 'by', 'ion', 'en', 'al', 'nt', 'hat', 'st', 'th', 'tion', 'me', 'll', 'her', 'le', 'ce', 'ts', 'that', 've', 'se', 'had', 'ut',  'are', 'not', 'ent', 'ch', 'k', 'w', 'ld', 'but', 'rs', 'ted', 'one', 'ere', 'ne', 'we', 'all', 'ns', 'ith', 'ad', 'ry', 'with', 'te', 'so', 'out', 'if', 'you', 'no', 'ay', 'ty']
         #common suffixes

    def fit(self, examples):
        return self

    def transform(self, examples):
        # in crease the count if the word ends in one of the common suffixes
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            words = nltk.word_tokenize(ex) # Split into list of words
            for word in words:
                for suffix in self.suffixes:
                    if word.lower().endswith(suffix):
                        features[i, 0] += 1
            features[i, 0] /= len(words)
            i += 1
        return features


class NGrams(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cv = CountVectorizer(analyzer='word',ngram_range=(1,2))

    def fit(self,examples):
        return self.cv.fit(examples)
    def transform(self,examples):
        return self.cv.transform(examples)

class Tfidf(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1,2))

    def fit(self,examples):
        return self.tfidf.fit(examples)
    def transform(self,examples):
        return self.tfidf.transform(examples)




class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),
            ('ngrams', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('n_grmas', NGrams())
            ])),
            ('tfidf', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', Tfidf())
            ])),
            ('suffix', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('suffix', Suffixes()),
            ])),
            ('pos_words', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('positive_words', pos_words()),
            ])),
            ('neg_words', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('negative_words', neg_words()),
            ])),
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    #print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=10000, shuffle=True, verbose=1)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)

    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier
