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
from keras.models import Model
from keras.regularizers import l1,l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=6, dim=(20,128,128), label_dim=(128,128), n_channels=3,
				 n_classes=2, shuffle=True):
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
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		print("Generating 1 batch...")
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		print("EPOCH END",self.indexes)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		Y = np.empty((self.batch_size, *self.label_dim), dtype=int)

		#print(list_IDs_temp)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample

			X[i,] = np.load('data/' + ID + '.npy')

			# Store class
			#print(ID)
			#print(np.load('labels/' + ID + '.npy')[-1].shape)
			Y[i,] = np.load('labels/' + ID + '.npy')[-1].astype(np.int)/255 # Only last frame is segmented

		#return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)
		return X, np.expand_dims(Y, axis=3)


def convolution_layer_over_time(x,filter_size,dilation_rate=1, kernel_size=3, weight_decay=1E-4):
	x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same'))(x)
	x = BatchNormalization(gamma_regularizer=l2(weight_decay),
						beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	return x
def convolution_layer(x,filter_size,dilation_rate=1, kernel_size=3, weight_decay=1E-4):
	x = Conv2D(filter_size, kernel_size, padding='same')(x)
	x = BatchNormalization(gamma_regularizer=l2(weight_decay),
						beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	return x
def transpose_layer(x,filter_size,dilation_rate=1, 
	kernel_size=3, strides=(2,2), weight_decay=1E-4):
	x = Conv2DTranspose(filter_size, 
		kernel_size, strides=strides, padding='same')(x)
	x = BatchNormalization(gamma_regularizer=l2(weight_decay),
										beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	return x	


def model_get(params):
	in_im = Input(shape=(*params['dim'], params['n_channels']))
	weight_decay=1E-4
	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=False,
		padding="same"))(in_im)
#	out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

	out = Conv2D(params['n_classes'], (1, 1), activation='softmax',
					padding='same')(x)

	#out = Activation('relu')(x)	

	model = Model(in_im, out)
	print(model.summary())
	return model
def model_get(params):
	in_im = Input(shape=(*params['dim'], params['n_channels']))
	weight_decay=1E-4

	fs=16

	p1=convolution_layer_over_time(in_im,fs)			
	p1=convolution_layer_over_time(p1,fs)
	e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
	p2=convolution_layer_over_time(e1,fs*2)
	e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
	p3=convolution_layer_over_time(e2,fs*4)
	e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=False,
			padding="same"),merge_mode='concat')(e3)

	d3 = transpose_layer(x,fs*4)
	#d3 = keras.layers.concatenate([d3, p3], axis=4)
	d3=convolution_layer(d3,fs*4)
	d2 = transpose_layer(d3,fs*2)
	#d2 = keras.layers.concatenate([d2, p2], axis=4)
	d2=convolution_layer(d2,fs*2)
	d1 = transpose_layer(d2,fs)
	#d1 = keras.layers.concatenate([d1, p1], axis=4)
	out=convolution_layer(d1,fs)
	out = Conv2D(1, (1, 1), activation='sigmoid',
								padding='same')(out)
	model = Model(in_im, out)
	print(model.summary())
	return model



def read_dict(path):
	'Reads Python dictionary stored in a csv file'
	dictionary = {}
	for key, val in csv.reader(open(path)):
		dictionary[key] = val
	return dictionary
def train_test_split(partition_train, validation_size=5):
	print(partition_train)
	train_len = len(partition_train)
	partition_train_new=partition_train[:validation_size]
	partition_validation=partition_train[validation_size:]
	return partition_train_new, partition_validation

def partition_get():
	partition={}
	sample_names = glob.glob(os.path.join('data/*.npy'))
	sample_names = [x.replace("data/","").replace(".npy","") for x in sample_names]
	print(sample_names)

	partition['train'] = sample_names[:30]
	partition['test'] = sample_names[30:]

	print(partition['train'],partition['test'])

	partition['train'], partition['validation'] = train_test_split(partition['train'], validation_size=5)

	return partition
if __name__ == "__main__":
	##---------------- Parameters --------------------------##
	im_len=128
	params = {
			'dim': (20,im_len,im_len),
			'label_dim': (im_len,im_len),
			'batch_size': 4,
			'n_classes': 2,
			'n_channels': 3,
			'shuffle': True}
	##---------------- Dataset -----------------------------##
	partition = partition_get()
	#pdb.set_trace()
	
	print("Train len: ",len(partition['train']), "Validation len: ",len(partition['validation']))
	print("partition['train']",partition['train'])
	print("partition['validation']",partition['validation'])
	#pdb.set_trace()
	##---------------- Model -------------------------------##
	model = model_get(params)

	##---------------- Train -------------------------------##

	training_generator = DataGenerator(partition['train'], partition['train'], **params)
	validation_generator = DataGenerator(partition['validation'], partition['validation'], **params)

	model.compile(optimizer=Adam(lr=0.01, decay=0.00016667),
					loss='binary_crossentropy',
					metrics=['accuracy'], options = run_opts)
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
	# Train model on dataset
	model.fit_generator(generator=training_generator,
						validation_data=validation_generator,
						use_multiprocessing=True,
						workers=9, 
						callbacks=[es])
