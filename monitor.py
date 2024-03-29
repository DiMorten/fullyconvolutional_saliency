
import numpy as np
from time import time
import numpy as np
import keras.backend as K
import keras
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv3D, Conv3DTranspose, AveragePooling3D
from keras.layers import AveragePooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, ConvLSTM2D
from keras.models import Model
from keras.layers import ELU, Lambda
from keras import layers
from keras import regularizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score
from sklearn.metrics import classification_report
from keras.callbacks import Callback
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
import pdb
from keras.regularizers import l1,l2
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, AveragePooling2D, Bidirectional, Activation
from icecream import ic
from pathlib import Path
import cv2
import joblib
elu_alpha = 0.1

def save_np_sample_as_png(sample, name_id=''):
    for frame_id in range(sample.shape[0]):
        frame = sample[frame_id]
        #ic(frame.shape)
        path = 'val_sample/'
        Path(path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path+'val_sample'+name_id+str(frame_id)+'.png', frame)


def scalerInverseApply(X, scaler):
    X_shape = X.shape
    X = np.reshape(X, (-1, X_shape[-1]))
	
    X = scaler.inverse_transform(X)
    X = np.reshape(X, X_shape)
    return X

class Monitor(Callback):
    def __init__(self, validation, patience, classes):   
        super(Monitor, self).__init__()
        self.validation = validation 
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.classes = classes
        self.f1_history = []
        self.oa_history = []
        #self.scaler = joblib.load('scaler.save')
        
    def on_train_begin(self, logs={}):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0

        
    def on_epoch_begin(self, epoch, logs={}):        
        self.pred = []
        self.targ = []

     
    def on_epoch_end(self, epoch, logs={}):
        #deb.prints(range(len(self.validation)))
        # num = np.random.randint(0,len(self.validation),1)
        for batch_index in range(len(self.validation)):
#            pdb.set_trace()
            val_targ = self.validation[batch_index][1]   
            val_pred = self.model.predict(self.validation[batch_index][0])
            val_input = self.validation[batch_index][0]
##            deb.prints(val_pred.shape) # was programmed to get two outputs> classif. and depth
##            deb.prints(val_targ.shape) # was programmed to get two outputs> classif. and depth
##            deb.prints(len(self.validation[batch_index][1])) # was programmed to get two outputs> classif. and depth


#            pdb.set_trace()


#            pdb.set_trace()

            val_predict = val_pred.argmax(axis=-1)            
#            val_targ = np.squeeze(val_targ)
            val_target = val_targ.argmax(axis=-1)

            self.pred.extend(val_predict.flatten())
            self.targ.extend(val_target.flatten())            

#        ic(self.pred.shape)
#        ic(self.targ.shape)
#        pdb.set_trace()
        f1 = np.round(f1_score(self.targ, self.pred, average=None)*100,2)
        precision = np.round(precision_score(self.targ, self.pred, average=None)*100,2)
        recall= np.round(recall_score(self.targ, self.pred, average=None)*100,2)
#        jac = np.round(jaccard_score(self.targ, self.pred, average=None)*100,2)

        #update the logs dictionary:
        mean_f1 = np.sum(f1)/self.classes
 #       mean_jac = np.sum(jac)/self.classes
        logs["mean_f1"]=mean_f1

        self.f1_history.append(mean_f1)
        
        print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
        print(f' — mean_f1: {mean_f1}')
#        print(f' — mean_jac: {mean_jac}')


        oa = np.round(accuracy_score(self.targ, self.pred)*100,2)
        print("oa",oa)        
        self.oa_history.append(oa)

        current = logs.get("mean_f1")
        if np.less(self.best, current):
            self.best = current
            self.wait = 0
            print("Found best weights at epoch {}".format(epoch + 1))
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

            #save_np_sample_as_png(val_targ[0].argmax(axis=-1).astype(np.uint8)*255, 'targ')
            #save_np_sample_as_png(val_pred[0].argmax(axis=-1).astype(np.uint8)*255, 'pred')
            #save_np_sample_as_png((val_input[0]*255.).astype(np.uint8), 'input')
###            save_np_sample_as_png((val_input[0, ..., :-1]*255.).astype(np.uint8), 'input')            
#            input_ = scalerInverseApply(val_input[0], self.scaler).astype(np.uint8)

#            save_np_sample_as_png(input_, 'input')


        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping".format(self.stopped_epoch + 1))
        print("f1 history",self.f1_history)
        print("oa history",self.oa_history)
        

