import cv2
import os
import glob
import numpy as np
import pdb
from pathlib import Path
import csv
import threading
import keras
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D
from keras.models import Model, load_model
from keras.regularizers import l1,l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from icecream import ic
import sys
from keras import backend as K
from keras import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib


class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=6, dim=(20,128,128), label_dim=(128,128), n_channels=3,
				 n_classes=2, shuffle=True, scaler=None):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		print("self.list_IDs",self.list_IDs)
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.label_dim = label_dim
		self.scaler = scaler
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
#		print("Generating 1 batch...")
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
#		print("EPOCH END",self.indexes)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def scalerApply(self, X):
		X_shape = X.shape
		X = np.reshape(X, (-1, X_shape[-1]))
		X = self.scaler.transform(X)
		X = np.reshape(X, X_shape)
		return X

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
#		Y = np.empty((self.batch_size, *self.label_dim, self.n_classes), dtype=int)
		Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

		#print(list_IDs_temp)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample

#			X[i,] = np.load('data/' + ID + '.npy').astype(np.float32)/255.0
			X[i,] = np.load('data/' + ID + '.npy').astype(np.float32)

#			ic(np.average(X[i]))
			##ic(X[i].shape)
			##cv2.imwrite("sample_input.png", X[i][-1].astype(np.int))

			# Y[i] is configured as N-to-1. Its shape is (h, w) 
			# For N-to-N config, delete the [-1] indexing to get all the label frames. 
			# 	That way, Y[i] shape will be (t, h, w)
			label = np.load('labels/' + ID + '.npy')
##			ic(np.unique(label, return_counts=True))
##			ic(np.unique(label[...,0], return_counts=True))
##			ic(np.unique(label[...,1], return_counts=True))

##			pdb.set_trace()
			#label = label[-1]
			##cv2.imwrite("sample_label.png", label[...,0].astype(np.int))
			##pdb.set_trace()

			Y[i] = label.astype(np.int)/255
		X = self.scalerApply(X)	
		#ic(np.average(X), np.std(X))	
		#pdb.set_trace()	


		return X, Y

