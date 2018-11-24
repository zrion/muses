from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os, ast, itertools
import pandas as pd
import matplotlib.pyplot as plt


# Evaluation helper
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, f1_score
from sklearn.metrics import make_scorer

# Training helper
from sklearn.model_selection import GridSearchCV			# Grid search for optimal params
import xgboost as xgb

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE					# Handle imbalanced set

# Metrics
from sklearn.metrics import accuracy_score
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


def k_fold_cv(X_train, y_train, num_folds, classifier):
	''' 
		k-fold cross validation, return the average error on classifying validation set
	'''
	# Call KFold to split the set
	kfold = KFold(n_splits=num_folds)

	sum_scores = 0
	# Training on each fold
	for k, (train, test) in enumerate(k_fold.split(X_train, y_train)):
		classifier.fit(X_train[train], y_train[train])
		scores = classifier.score(X_train[test], y_train[test])
		print("[fold {0}], score: {2:.5f}".format(k, scores))

		sum_scores += scores

	avg_scores = float(sum_scores)/num_folds

	return avg_scores

def evaluation(y_test, y_pred):
	confusion_matrix = confusion_matrix(y_test, y_pred) 
	accuracy = 	accuracy_score(y_test,y_pred)*100
	
	print ("Accuracy:", accuracy) 
	  
	return accuracy

def modelfit_XGB(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	'''
		Fitting function for our main method: XGBoost.
		Thanks https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
	'''
	if useTrainCV:
		xgb_param 	= alg.get_xgb_params()
		xgb_param['num_class'] = 16
		xgtrain 	= xgb.DMatrix(X_train, label=y_train)
		cvresult 	= xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
		    metrics='mlogloss', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
		print (cvresult)
		alg.set_params(n_estimators=cvresult.shape[0])

	# Fit the algorithm on the data
	alg.fit(X_train, y_train, eval_metric='mlogloss')
	    
	# Predict training set:
	y_pred = alg.predict(X_train)
	y_predprob = alg.predict_proba(X_train)
	    
	# Print model report:
	print ("\nModel Report")
	print ("Accuracy : %.4g" % accuracy_score(y_train, y_pred))
	print ("Log loss (Train): %f" % log_loss(y_train, y_predprob))

	# feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	# feat_imp.plot(kind='bar', title='Feature Importances')
	# plt.ylabel('Feature Importance Score')

	return

def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

# Function to make predictions 
def prediction(X_test, clf_object): 
	y_pred = clf_object.predict(X_test)
	return y_pred

# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
	accuracy = 	accuracy_score(y_test,y_pred)*100
	print ("Accuracy:", accuracy) 
	return accuracy