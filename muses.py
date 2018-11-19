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
		'n_estimators': [100, 500, 1000, 2000, 3000],
		'max_depth': range(3, 18, 2)
	}

	# file = dirname(sys.argv[0]) + "/results/extra_trees_training_result.txt"
	# f = open(file, 'w')

	# scoring = {'Balanced_accuracy': make_scorer(balanced_accuracy_score), 'Accuracy': make_scorer(accuracy_score)}

	# estimator = ExtraTreesClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
	# gsearch= GridSearchCV(estimator = estimator, param_grid = param_extratrees, scoring=scoring,refit='Balanced_accuracy',n_jobs=-1,iid=False, cv=5, verbose=100)
	# gsearch.fit(X_train, y_train)
	# print (gsearch.cv_results_)
	# print (gsearch.best_params_)
	# print (gsearch.best_score_)
	# print (gsearch.best_estimator_)

	# f.write(str(gsearch.cv_results_))
	# f.write(str(gsearch.best_params_))
	# f.write(str(gsearch.best_score_))
	# f.write(str(gsearch.best_estimator_))

	# f.close()

	# For softmax regression
	# Softmax param
	param_softmax = {
		'penalty': ['l1', 'l2'],
		'C': [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]   				# Inverse of lambda
	}

	file = dirname(sys.argv[0]) + "/results/softmax_training_result_all_features.txt"
	f = open(file, 'w')

	scoring = {'Balanced_accuracy': make_scorer(balanced_accuracy_score), 'Accuracy': make_scorer(accuracy_score)}

	estimator = LogisticRegression(penalty='l2', C=1, solver='sag', multi_class='multinomial')
	gsearch= GridSearchCV(estimator = estimator, param_grid = param_softmax, scoring=scoring,refit='Balanced_accuracy',n_jobs=-1,iid=False, cv=5, verbose=10)
	gsearch.fit(X_train, y_train)
	print (gsearch.cv_results_)
	print (gsearch.best_params_)
	print (gsearch.best_score_)
	print (gsearch.best_estimator_)
	f.write(str(gsearch.cv_results_) + "\n")
	f.write(str(gsearch.best_params_) + "\n")
	f.write(str(gsearch.best_score_) + "\n")
	f.write(str(gsearch.best_estimator_))
	f.close()

	print(">..........................................................")
	# For svm RBF, linear
	param_SVM = {
		'kernel': ['rbf', 'linear'],
		'C': [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]   				# Inverse of lambda
	}

	file = dirname(sys.argv[0]) + "/results/SVM_training_result_all_features.txt"
	f = open(file, 'w')

	scoring = {'Balanced_accuracy': make_scorer(balanced_accuracy_score), 'Accuracy': make_scorer(accuracy_score)}

	estimator = SVC(C=1, kernel='rbf', tol=1e-4)
	gsearch= GridSearchCV(estimator = estimator, param_grid = param_SVM, scoring=scoring,refit='Balanced_accuracy',n_jobs=-1,iid=False, cv=5, verbose=10)
	gsearch.fit(X_train, y_train)
	print (gsearch.cv_results_)
	print (gsearch.best_params_)
	print (gsearch.best_score_)
	print (gsearch.best_estimator_)
	f.write(str(gsearch.cv_results_) + "\n")
	f.write(str(gsearch.best_params_) + "\n")
	f.write(str(gsearch.best_score_) + "\n")
	f.write(str(gsearch.best_estimator_))
	f.close()

	print(">..........................................................")

	# For neural network, 1 hidden layer
	param_nn_1 = {
		'hidden_layer_sizes': [(100, ), (150, ), (200, )]
		'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
		'activation':['relu', 'tanh', 'logistic', 'identity']  				# Inverse of lambda
	}

	file = dirname(sys.argv[0]) + "/results/neural_network_1_training_result_all_features.txt"
	f = open(file, 'w')

	scoring = {'Balanced_accuracy': make_scorer(balanced_accuracy_score), 'Accuracy': make_scorer(accuracy_score)}

	estimator = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.0001, learning_rate_init=1e-4, verbose=True) # Batch size auto: 200 samples
	gsearch= GridSearchCV(estimator = estimator, param_grid = param_nn_1, scoring=scoring,refit='Balanced_accuracy',n_jobs=-1,iid=False, cv=5, verbose=10)
	gsearch.fit(X_train, y_train)
	print (gsearch.cv_results_)
	print (gsearch.best_params_)
	print (gsearch.best_score_)
	print (gsearch.best_estimator_)
	f.write(str(gsearch.cv_results_) + "\n")
	f.write(str(gsearch.best_params_) + "\n")
	f.write(str(gsearch.best_score_) + "\n")
	f.write(str(gsearch.best_estimator_))
	f.close()

	print(">..........................................................")

	# For neural network, 2 hidden layer
	param_nn_2 = {
		'hidden_layer_sizes': [(100, 100), (150, 150), (200, 200), (150, 200), (100, 150)]
		'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
		'activation':['relu', 'tanh', 'logistic', 'identity']  				# Inverse of lambda
	}

	file = dirname(sys.argv[0]) + "/results/neural_network_2_training_result_all_features.txt"
	f = open(file, 'w')

	scoring = {'Balanced_accuracy': make_scorer(balanced_accuracy_score), 'Accuracy': make_scorer(accuracy_score)}

	estimator = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu', alpha=0.0001, learning_rate_init=1e-4, verbose=True) # Batch size auto: 200 samples
	gsearch= GridSearchCV(estimator = estimator, param_grid = param_nn_2, scoring=scoring,refit='Balanced_accuracy',n_jobs=-1,iid=False, cv=5, verbose=10)
	gsearch.fit(X_train, y_train)
	print (gsearch.cv_results_)
	print (gsearch.best_params_)
	print (gsearch.best_score_)
	print (gsearch.best_estimator_)
	f.write(str(gsearch.cv_results_) + "\n")
	f.write(str(gsearch.best_params_) + "\n")
	f.write(str(gsearch.best_score_) + "\n")
	f.write(str(gsearch.best_estimator_))
	f.close()

	print(">..........................................................")

	# For decision tree
	param_dt = {
		'max_depth': range(5, 18, 2)
	}

	file = dirname(sys.argv[0]) + "/results/decision_tree_training_result_all_features.txt"
	f = open(file, 'w')

	scoring = {'Balanced_accuracy': make_scorer(balanced_accuracy_score), 'Accuracy': make_scorer(accuracy_score)}

	estimator = DecisionTreeClassifier(max_depth=5)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			(hidden_layer_sizes=(100,100), activation='relu', alpha=0.0001, learning_rate_init=1e-4, verbose=True) # Batch size auto: 200 samples
	gsearch= GridSearchCV(estimator = estimator, param_grid = param_dt, scoring=scoring,refit='Balanced_accuracy',n_jobs=-1,iid=False, cv=5, verbose=10)
	gsearch.fit(X_train, y_train)
	print (gsearch.cv_results_)																																																											
	print (gsearch.best_params_)
	print (gsearch.best_score_)
	print (gsearch.best_estimator_)
	f.write(str(gsearch.cv_results_) + "\n")
	f.write(str(gsearch.best_params_) + "\n")
	f.write(str(gsearch.best_score_) + "\n")																																																																																																																																																																																																																																						
	f.write(str(gsearch.best_estimator_))
	f.close()																			

	return


if __name__ == '__main__':
	multiprocessing.set_start_method('forkserver')
	main()

