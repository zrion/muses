from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os, ast
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

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