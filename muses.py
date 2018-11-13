#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utilities import *

import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import ast
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

def main():

	features_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/features.csv"
	tracks_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/tracks.csv"
	genres_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/genres.csv"

	# Load dataset
	print "Load dataset..."
	tracks = load(tracks_file)
	genres = load(genres_file)
	features = load(features_file)

	# Parse medium set
	medium = tracks['set', 'subset'] <= 'medium'

	# Split dataset
	train = tracks['set', 'split'] == 'training'
	val = tracks['set', 'split'] == 'validation'
	test = tracks['set', 'split'] == 'test'

	y_train = tracks.loc[medium & train, ('track', 'genre_top')]
	y_test = tracks.loc[medium & test, ('track', 'genre_top')]
	enc = skl.preprocessing.LabelEncoder()
	y_train = enc.fit_transform(y_train)
	y_test = enc.transform(y_test)

	X_train = features.loc[medium & train, 'mfcc']
	X_test = features.loc[medium & test, 'mfcc']

	print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
	print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

	# Be sure training samples are shuffled.
	X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

	# Standardize features by removing the mean and scaling to unit variance.
	scaler = skl.preprocessing.StandardScaler(copy=False)
	scaler.fit_transform(X_train)
	scaler.transform(X_test)

	# Support vector classification.
	# clf = skl.svm.SVC()

	# First XGBoost
	clf = XGBClassifier(n_estimator=1000, max_depth=5, objective='multi:softmax')

	# First ExtraTrees
	# clf = ExtraTreesClassifier(n_estimator=2000)

	print "Start training..."
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score = accuracy_score(y_test, y_pred)
	print('Accuracy: {:.2%}'.format(score))

	
	return


if __name__ == '__main__':
	main()
	
