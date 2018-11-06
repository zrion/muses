#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, dirname, basename, splitext, realpath
import sys, re, os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import utils
import ast


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