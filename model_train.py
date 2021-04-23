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

from monitor import Monitor
from datagenerator import DataGenerator
#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = False)
def load_labels(partition, params):
	samples = np.zeros((len(partition), *params['dim'] , params['n_classes']))
	for idx, filename in enumerate(partition):
		#ic(filename)
		samples[idx] = np.load('labels/' + filename +'.npy')
	return samples

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#	return f1_score(y_true, y_pred, average='macro')

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels

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
def transpose_layer_over_time(x,filter_size,dilation_rate=1, 
	kernel_size=3, strides=(2,2), weight_decay=1E-4):
	x = TimeDistributed(Conv2DTranspose(filter_size, 
		kernel_size, strides=strides, padding='same'))(x)
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


	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=False,
			padding="same"),merge_mode='concat')(in_im)

	out = Conv2D(1, (1, 1), activation='sigmoid',
								padding='same')(x)
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
	out = Conv2D(params['n_classes'], (1, 1), activation='sigmoid',
								padding='same')(out)
	model = Model(in_im, out)
	print(model.summary())
	return model


def BUnetConvLSTM_NtoN(params):
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

	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
			padding="same"),merge_mode='concat')(e3)
#	x = TimeDistributed(Conv2D(64, (1, 1), activation='relu',
#								padding='same'))(e3)
	d3 = transpose_layer_over_time(x,fs*4)
	d3 = keras.layers.concatenate([d3, p3], axis=-1)
	d3=convolution_layer_over_time(d3,fs*4)
	d2 = transpose_layer_over_time(d3,fs*2)
	d2 = keras.layers.concatenate([d2, p2], axis=-1)
	d2=convolution_layer_over_time(d2,fs*2)
	d1 = transpose_layer_over_time(d2,fs)
	d1 = keras.layers.concatenate([d1, p1], axis=-1)
	out=convolution_layer_over_time(d1,fs)
	out = TimeDistributed(Conv2D(params['n_classes'], (1, 1), activation='softmax',
								padding='same'))(out)
	model = Model(in_im, out)
	print(model.summary())
	return model



def BUnetConvLSTM_Skip_NtoN(params):
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

	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
			padding="same"),merge_mode='concat')(e3)

	d3 = transpose_layer_over_time(x,fs*4)
	d3 = keras.layers.concatenate([d3, p3], axis=-1)
	d3=convolution_layer_over_time(d3,fs*4)
	d2 = transpose_layer_over_time(d3,fs*2)
	d2 = keras.layers.concatenate([d2, p2], axis=-1)
	d2=convolution_layer_over_time(d2,fs*2)
	d1 = transpose_layer_over_time(d2,fs)
	d1 = keras.layers.concatenate([d1, p1], axis=-1)
	out=convolution_layer_over_time(d1,fs)
	out = TimeDistributed(Conv2D(params['n_classes'], (1, 1), activation='softmax',
								padding='same'))(out)
	model = Model(in_im, out)
	print(model.summary())
	return model


def BUnetConvLSTM_Skip_NtoN(params):
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

	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
			padding="same"),merge_mode='concat')(e3)

	d3 = transpose_layer_over_time(x,fs*4)
	d3 = keras.layers.concatenate([d3, p3], axis=-1)
	d3=convolution_layer_over_time(d3,fs*4)
	d2 = transpose_layer_over_time(d3,fs*2)
	d2 = keras.layers.concatenate([d2, p2], axis=-1)
	d2=convolution_layer_over_time(d2,fs*2)
	d1 = transpose_layer_over_time(d2,fs)
	d1 = keras.layers.concatenate([d1, p1], axis=-1)
	out=convolution_layer_over_time(d1,fs)
	out = TimeDistributed(Conv2D(params['n_classes'], (1, 1), activation='sigmoid',
								padding='same'))(out)
	model = Model(in_im, out)
	print(model.summary())
	return model
def BConvLSTM_NtoN(params):
	in_im = Input(shape=(*params['dim'], params['n_channels']))
	weight_decay=1E-4

	fs=16


	x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
			padding="same"),merge_mode='concat')(in_im)

	out = TimeDistributed(Conv2D(params['n_classes'], (1, 1), activation='sigmoid',
								padding='same'))(x)
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
	partition_validation=partition_train[:validation_size]
	partition_train_new=partition_train[validation_size:]
	return partition_train_new, partition_validation

def partition_get():
	partition={}


	partition['train'] = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'drift-turn', 'elephant', 'flamingo',
    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
    'rhino', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']


	partition['test'] = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
    'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black',
    'soapbox']


	ic(partition['train'])
	ic(partition['test'])

	partition['train'], partition['validation'] = train_test_split(partition['train'], validation_size=8)
	ic(partition['validation'])

	return partition

if __name__ == "__main__":
	##---------------- Parameters --------------------------##
	im_len=128
	t_len = 20 # it may be 20 later
	scaler = joblib.load('scaler.save')
	params = {
			'dim': (t_len,im_len,im_len),
			'label_dim': (im_len,im_len),
			'batch_size': 4,
			'n_classes': 2,
#			'n_channels': 3,
			'n_channels': 4,

			'shuffle': True,
			'scaler' : scaler}
	paramsTest = params.copy()
	paramsTest['batch_size']=1
	paramsTest['shuffle'] = False
	##---------------- Dataset -----------------------------##
	partition = partition_get()
#	class_weights = np.array([0.54569158, 5.97146725])
	class_weights = np.array([0.53869264, 6.96117691]) # all is 1
#	class_weights = np.array([0.54175429, 6.48740847]) # all is 1
	class_weights = np.array([0.54248901, 6.38387407]).astype(np.float32)

	
	#pdb.set_trace()
	
	print("Train len: ",len(partition['train']), "Validation len: ",len(partition['validation']))
	print("partition['train']",partition['train'])
	print("partition['validation']",partition['validation'])
	#pdb.set_trace()
	##---------------- Model -------------------------------##
	model = BUnetConvLSTM_NtoN(params)


	##---------------- Train -------------------------------##
	ic(partition['test'])
	training_generator = DataGenerator(partition['train'], partition['train'], **params)
	validation_generator = DataGenerator(partition['validation'], partition['validation'], **params)
	test_generator = DataGenerator(partition['test'], partition['test'], **paramsTest)

	
	file_output='model.hdf5'
	trainMode = False
	if trainMode == True:
		model.compile(optimizer=Adam(lr=0.0001, decay=0.00016667),
					#loss='binary_crossentropy',
					loss=weighted_categorical_crossentropy(class_weights),
#					loss=categorical_focal_loss(gamma=2., alpha=.25),
#					metrics=['accuracy', mean_iou])
#					metrics=['accuracy', mean_iou])
					metrics=['accuracy'])
					

#		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, min_delta=0.001)
		es = EarlyStopping(monitor='val_f1_m', mode='max', verbose=1, patience=15, min_delta=0.001)
#		es = EarlyStopping(monitor='val_mean_iou', mode='max', verbose=1, patience=15, min_delta=0.001)

		# Train model on dataset
		history = model.fit_generator(generator=training_generator,
							epochs=500,
							validation_data=validation_generator,
#							callbacks=[es])
							callbacks = [Monitor(validation=validation_generator,patience = 15,
												classes=2)]) 
		model.save(file_output)
		metrics_evaluated = model.evaluate_generator(test_generator)
		print('Test accuracy :', metrics_evaluated)
	else:
		model = load_model(file_output, compile=False)



	predictions = model.predict_generator(test_generator)
	labels = load_labels(partition['test'], params)
	
	ic(labels.shape)
	ic(predictions.shape)
	# get metrics
	labels_flat = labels.argmax(axis=-1).flatten()
	predictions_flat = predictions.argmax(axis=-1).flatten()
	report = classification_report(labels_flat,predictions_flat)
	print(report)
	print(confusion_matrix(labels_flat,predictions_flat))
	ic(np.unique(predictions, return_counts=True))
	ic(np.unique(predictions_flat, return_counts=True))

	# save to img
	
	for idx, name in enumerate(partition['test']):
		for frame_id in range(t_len):
			frame = predictions.argmax(axis=-1)[idx, frame_id].astype(np.uint8)*255
			#ic(frame.shape)
			path = 'sample/prediction/'
			Path(path).mkdir(parents=True, exist_ok=True)
			cv2.imwrite(path+name+str(frame_id)+'.png', frame)
	for idx, name in enumerate(partition['test']):
		for frame_id in range(t_len):
			frame = labels.argmax(axis=-1)[idx, frame_id].astype(np.uint8)*255
			#ic(frame.shape)
			path = 'sample/label/'
			Path(path).mkdir(parents=True, exist_ok=True)			
			cv2.imwrite(path+name+str(frame_id)+'.png', frame)
	pdb.set_trace()