#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import ast
import xgboost as xgb


def load(filepath):

		filename = os.path.basename(filepath)

		if 'features' in filename:
			return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

		if 'echonest' in filename:
			return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

		if 'genres' in filename:
			return pd.read_csv(filepath, index_col=0)

		if 'tracks' in filename:
			tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

		COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
					('track', 'genres'), ('track', 'genres_all')]
		for column in COLUMNS:
			tracks[column] = tracks[column].map(ast.literal_eval)

		COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
					('album', 'date_created'), ('album', 'date_released'),
					('artist', 'date_created'), ('artist', 'active_year_begin'),
					('artist', 'active_year_end')]
		for column in COLUMNS:
			tracks[column] = pd.to_datetime(tracks[column])

		SUBSETS = ('small', 'medium', 'large')
		tracks['set', 'subset'] = tracks['set', 'subset'].astype(
			'category', categories=SUBSETS, ordered=True)

		COLUMNS = [('track', 'license'), ('artist', 'bio'),
					('album', 'type'), ('album', 'information')]

		for column in COLUMNS:
			tracks[column] = tracks[column].astype('category')

		return tracks

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

	# # Support vector classification.
	# clf = skl.svm.SVC()
	# print "Start training..."
	# clf.fit(X_train, y_train)
	# score = clf.score(X_test, y_test)
	# print('Accuracy: {:.2%}'.format(score))
	dtrain = xgb.DMatrix(X_train, y_train)
	dtest  = xgb.DMatrix(X_test, y_test)

	return


if __name__ == '__main__':
	main()
	
# features_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/features.csv"
# tracks_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/tracks.csv"
# genres_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/genres.csv"


# tracks = load(tracks_file)
# genres = load(genres_file)
# features = load(features_file)

# medium = tracks['set', 'subset'] <= 'medium'

# train = tracks['set', 'split'] == 'training'
# val = tracks['set', 'split'] == 'validation'
# test = tracks['set', 'split'] == 'test'

# print medium




















# def main():
# 	return

# if __name__ == "__main__":
# 	main()