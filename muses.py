#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utilities import *
from model import *
import multiprocessing

def main():

	features_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/features.csv"
	tracks_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/tracks.csv"
	genres_file = dirname(realpath(sys.argv[0])) + "/fma_dataset/genres.csv"

	# Load dataset
	print ("Load dataset...")
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
	enc = LabelEncoder()
	y_train = enc.fit_transform(y_train)
	y_test = enc.transform(y_test)

	X_train = features.loc[medium & train]
	X_test = features.loc[medium & test]

	print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
	print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

	# Be sure training samples are shuffled.
	X_train, y_train = shuffle(X_train, y_train, random_state=42)

	# Standardize features by removing the mean and scaling to unit variance.
	scaler = StandardScaler(copy=False)
	scaler.fit_transform(X_train)
	scaler.transform(X_test)

	# Support vector classification.
	# clf = skl.svm.SVC()

	# Tuning n_estimators
	# xgb1 = XGBClassifier(
	# 	 learning_rate =0.1,
	# 	 n_estimators=1000,
	# 	 max_depth=5,
	# 	 min_child_weight=1,
	# 	 gamma=0,
	# 	 subsample=0.8,
	# 	 colsample_bytree=0.8,
	# 	 objective= 'multi:softmax',
	# 	 nthread=4,
	# 	 scale_pos_weight=1,
	# 	 seed=50)

	# modelfit_XGB(xgb1, X_train, y_train)

	# First ExtraTrees
	# clf = ExtraTreesClassifier(n_estimator=2000)

	# print "Start training..."
	# clf.fit(X_train, y_train)


	# y_pred = clf.predict(X_test)
	# score = accuracy_score(y_test, y_pred)
	# print('Accuracy: {:.2%}'.format(score))

	# param_test1 = {
	#  'max_depth':range(3,10,2),
	#  'min_child_weight':range(1,6,2)
	# }

	param_test2 = {
	 'max_depth':[4,5,6],
	 'min_child_weight':[4,5,6]
	}
	gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=229, max_depth=5,
	 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
	 objective= 'multi_softmax', scale_pos_weight=1, seed=50), 
	 param_grid = param_test2, scoring='neg_log_loss',n_jobs=-1,iid=False, cv=5, verbose=100)
	gsearch1.fit(X_train, y_train)
	print (gsearch1.cv_results_)
	print (gsearch1.best_params_)
	print (gsearch1.best_score_)	


	return


if __name__ == '__main__':
	multiprocessing.set_start_method('forkserver')
	main()

