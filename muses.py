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

	# XGBClassifier(learning_rate =0.1, n_estimators=229, max_depth=5,
	#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
	#  objective= 'multi_softmax', scale_pos_weight=1, seed=50)

	param_test2 = {
	 'max_depth':[4,5,6],
	 'min_child_weight':[4,5,6]
	}

	# ExtraTrees param
	param_extratrees = {
		'n_estimators': [100, 300, 500, 1000, 2000],
		# 'max_depth': [3, 5, 7, 9]
	}
	file = dirname(sys.argv[0]) + "/results/extra_trees_n_estimators_depth_5.txt"
	f = open(file, 'w') 
	# For extratrees
	for n_estimators in param_extratrees['n_estimators']:
		clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=5, n_jobs=-1)
		clf.fit(X_train, y_train)
		y_pred_train = clf.predict(X_train)
		y_pred_test = clf.predict(X_test)

		train_accuracy = accuracy_score(y_train, y_pred_train)*100
		train_balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)*100
		train_f1_score = f1_score(y_train, y_pred_train, average='micro')
		test_accuracy = accuracy_score(y_test, y_pred_test)*100
		test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)*100
		test_f1_score = f1_score(y_test, y_pred_test, average='micro')

		print ("Result for n_estimator", n_estimators, "with", 	X_train.shape[1], "features" )
		print ("Training accuracy:", str(train_accuracy))
		print ("Training balanced accuracy:", str(train_balanced_accuracy))
		print ("Training F1 score:",	 str(train_f1_score))
		print ("Test accuracy:", str(test_accuracy))
		print ("Test balanced accuracy:", str(test_balanced_accuracy))
		print ("Test F1 score:", str(test_f1_score))

		f.write("Result for n_estimator " + str(n_estimators) + " with " + str(X_train.shape[1]) + " features\n")
		f.write("Training accuracy: "+ str(train_accuracy)+"\n")
		f.write("Training balanced accuracy: " + str(train_balanced_accuracy)+"\n")
		f.write("Training F1 score: " + str(train_f1_score)+"\n")
		f.write("Test accuracy:" + str(test_accuracy)+"\n")
		f.write("Test balanced accuracy: " + str(test_balanced_accuracy)+"\n")
		f.write("Test F1 score: " + str(test_f1_score)+"\n")

		f.write(">-------------------------------------------------<\n")

	return


if __name__ == '__main__':
	multiprocessing.set_start_method('forkserver')
	main()

