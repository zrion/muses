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

	# Best models:
	best_XGB = XGBClassifier()
	best_DT = DecisionTreeClassifier()
	best_NN = MLPClassifier()
	best_LR = LogisticRegression()
	best_SVM = SVC()
	best_ET = ExtraTreesClassfier()