#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os, json, copy, time
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
	file = dirname(realpath(sys.argv[0])) + "/results/test_results_all_models_with_PCA.txt"
	recording = open(file, 'w')

	# Best models:
	best_XGB = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=9, n_jobs=-1,
	 min_child_weight=10, gamma=0.3, subsample=0.7, colsample_bytree=0.8, max_delta_step=1,
	 objective= 'multi_softmax', scale_pos_weight=1, reg_lambda=1, seed=50)
	best_SVM_RBF = SVC(C=500, kernel='rbf', tol=1e-4)
	best_DT = DecisionTreeClassifier(max_depth=21)
	best_NN1 = MLPClassifier(hidden_layer_sizes=(600,), activation='relu', alpha=0.1, learning_rate_init=1e-4, verbose=True)
	best_NN2 = MLPClassifier(hidden_layer_sizes=(600,100), activation='relu', alpha=0.01, learning_rate_init=1e-4, verbose=True)
	best_LR = SGDClassifier(loss='log', penalty='l1', alpha=0.0001, max_iter=5000, tol=1e-4)
	best_SVM_linear = SVC(C=0.1, kernel='linear', tol=1e-4)
	best_ET = ExtraTreesClassifier(n_estimators=3000, max_depth=17, n_jobs=-1)

	# PCA
	PCA_dims = [150, 225, 300, 375, 450]
	# Test phase
	for dimension in PCA_dims:
		print("Using PCA with # of dimensions:", dimension)
		currenttime = time.time()
		X_train_to_PCA = copy.deepcopy(X_train)
		X_test_to_PCA = copy.deepcopy(X_test)
		pca = PCA(n_components=dimension)
		X_train_PCA = pca.fit_transform(X_train_to_PCA)
		X_test_PCA = pca.fit_transform(X_test_to_PCA)

		print("Calculate accuracies and confusion matrices for models...")
		print("Fitting XGB...")
		accuracy_xgb, bl_accuracy_xgb = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_XGB, enc, title='Confusion matrix for XGBoost', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/XGB_" + str(dimension) + "_dims.png")	
		print("Fitting SVM RBF...")
		accuracy_svm_rbf, bl_accuracy_svm_rbf = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_SVM_RBF, enc, title='Confusion matrix for SVM RBF', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/SVM_RBF_" + str(dimension) + "_dims.png")
		print("Fitting DT...")
		accuracy_dt, bl_accuracy_dt = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_DT, enc, title='Confusion matrix for Decision Tree', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/decision_tree_" + str(dimension) + "_dims.png")
		print("Fitting NN1...")
		accuracy_nn1, bl_accuracy_nn1 = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_NN1, enc, title='Confusion matrix for Neural Network 3 layers', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/neural_network1_" + str(dimension) + "_dims.png")
		print("Fitting NN2...")
		accuracy_nn2, bl_accuracy_nn2 = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_NN2, enc, title='Confusion matrix for Neural Network 4 layers', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/neural_network2_" + str(dimension) + "_dims.png")
		print("Fitting LR...")
		accuracy_lr, bl_accuracy_lr = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_LR, enc, title='Confusion matrix for Logistic Regression', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/logistic_regression_" + str(dimension) + "_dims.png")
		print("Fitting SVM Linear...")
		accuracy_svm_linear, bl_accuracy_svm_linear = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_SVM_linear, enc, title='Confusion matrix for SVM Linear', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/svm_linear_" + str(dimension) + "_dims.png")
		print("Fitting ExtraTrees...")
		accuracy_et, bl_accuracy_et = testing_model(X_train_PCA, y_train, X_test_PCA, y_test, best_ET, enc, title='Confusion matrix for Extra Trees', savefig = dirname(realpath(sys.argv[0])) + "/results/conf_matrix/extratrees_" + str(dimension) + "_dims.png")

		print("Saving records...")
		print("Decision Tree:", "Accuracy", accuracy_dt, "Balanced accuracy", bl_accuracy_dt)
		print("Neural Network 1:", "Accuracy", accuracy_nn1, "Balanced accuracy", bl_accuracy_nn1)
		print("Neural Network 2:", "Accuracy", accuracy_nn2, "Balanced accuracy", bl_accuracy_nn2)
		print("Logistic Regression (OVR):", "Accuracy", accuracy_lr, "Balanced accuracy", bl_accuracy_lr)
		print("SVM linear:", "Accuracy", accuracy_svm_linear, "Balanced accuracy", bl_accuracy_svm_linear)
		print("ExtraTrees:", "Accuracy", accuracy_et, "Balanced accuracy", bl_accuracy_et)
		print("XGBoost:", "Accuracy", accuracy_xgb, "Balanced accuracy", bl_accuracy_xgb)
		print("SVM RBF:", "Accuracy", accuracy_svm_rbf, "Balanced accuracy", bl_accuracy_svm_rbf)
		print(">--------------------------------------------------------------")

		recording.write("Using PCA with dimension: " + str(dimension) + "\n")
		recording.write("XGBoost: " + " Accuracy " + str(accuracy_xgb) + " Balanced accuracy " + str(bl_accuracy_xgb) + "\n")
		recording.write("SVM RBF: " + " Accuracy " + str(accuracy_svm_rbf) + " Balanced accuracy " + str(bl_accuracy_svm_rbf) + "\n")	
		recording.write("Decision Tree: " + " Accuracy " + str(accuracy_dt) + " Balanced accuracy " + str(bl_accuracy_dt) + "\n")
		recording.write("Neural Network 1: " + " Accuracy " + str(accuracy_nn1) + " Balanced accuracy " + str(bl_accuracy_nn1) + "\n")
		recording.write("Neural Network 2: " + " Accuracy " + str(accuracy_nn2) + " Balanced accuracy " + str(bl_accuracy_nn2) + "\n")
		recording.write("Logistic Regression (OVR): " + " Accuracy " + str(accuracy_lr) + " Balanced accuracy " + str(bl_accuracy_lr) + "\n")
		recording.write("SVM Linear: " + " Accuracy " + str(accuracy_svm_linear) + " Balanced accuracy " + str(bl_accuracy_svm_linear) + "\n")
		recording.write("ExtraTrees: " + " Accuracy " + str(accuracy_et) + " Balanced accuracy " + str(bl_accuracy_et) + "\n")
		recording.write(">------------------------------------------------\n")

		print("Total processing time:", time.time()-currenttime)
	recording.close()		

if __name__ == "__main__":
	main()
