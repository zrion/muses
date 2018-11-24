#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os, json
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

	# Recording file
	file = dirname(realpath(sys.argv[0])) + "/results/test_results_all_features_except_XGB.txt"
	recording = open(file, 'w')

	# Best models:
	# best_XGB = XGBClassifier()
	best_DT = DecisionTreeClassifier(max_depth=21)
	best_NN1 = MLPClassifier(hidden_layer_sizes=(600,), activation='relu', alpha=0.1, learning_rate_init=1e-4, verbose=True)
	best_NN2 = MLPClassifier(hidden_layer_sizes=(600,100), activation='relu', alpha=0.01, learning_rate_init=1e-4, verbose=True)
	best_LR = LogisticRegression(loss='log', penalty='l1', alpha=0.0001, max_iter=5000, tol=1e-4)
	best_SVM = SVC(C=0.1, kernel='linear', tol=1e-4)
	best_ET = ExtraTreesClassfier(n_estimators=3000, max_depth=17, n_jobs=-1)

	# Fitting and testing each model
	best_DT.fit(X_train, y_train)
	y_pred_dt = best_DT.predict(X_test)

	best_NN1.fit(X_train, y_train)
	y_pred_nn1 = best_NN1.predict(X_test)

	best_NN2.fit(X_train, y_train)
	y_pred_nn2 = best_NN2.predict(X_test)

	best_LR.fit(X_train, y_train)
	y_pred_lr = best_LR.predict(X_test)

	best_SVM.fit(X_train, y_train)
	y_pred_svm = best_SVM.predict(X_test)

	best_ET.fit(X_train, y_train)
	y_pred_et = best_ET.predict(X_test)