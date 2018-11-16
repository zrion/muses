import xgboost as xgb                               	
from xgboost import XGBClassifier						# XGBoost
from sklearn.ensemble import ExtraTreesClassifier 		# ExtraTrees
from sklearn.neural_network import MLPClassifier		# Multilayer Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier			
from sklearn.svm import SVC

import pandas as pd
import numpy as np


def ExtraTrees(n_estimators, max_depth):
	'''
		Extra trees classifier: We train with different values of n_estimator: [100, 500, 1000, 2000, 3000] and max_depth [3, 5, None]
	'''
	return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth)

def XGBoost(n_estimators, max_depth):
	'''
		XGBoost classifier: We train with different values of n_estimator: [1000, 2000, 3000] and max_depth [3, 4, 5]
	'''
	return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
		objective='multi:softmax', silent=False, max_delta_step=1)

def DNN(n_hl, optimizer, activation):
	'''
		DNN Classifier: n_hl fixed (maybe 3 or 4), optimizer: ['sgd', 'adam'], activation: ['relu', 'logistic']
	'''
	return 

def Logistic():
	'''
		Softmax regression. Considering implementing from scratch.
	'''
	return

def SVM(kernel):
	'''
		SVM. Considering implementing from scratch. Will use [linear, RBF] kernels
	'''
	return

def DTree(max_depth):
	'''
		Decision tree classifier. Considering implementing from scratch. Will change max_depth
	'''
	return
