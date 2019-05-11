#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:18:23 2016

@author: renato
"""


#--------------------------------------------------------------
from Trading.Training.TradingModel import TradingModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2, activity_l2

import theano.tensor as T

from Framework.Miscellaneous.my_utils import *

modelpath = './models/weights'
modelname = 'G10-testV0'

epsilon = 1.0e-9
alpha = 0.01
_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    #out = -K.dot(y_true, y_pred)
    #out = -K.mean(K.minimum(y_true * y_pred, 5), axis=-1)
    out = -K.mean(y_true * y_pred, axis=-1)
    #out2 = K.mean(y_pred**2)
    #out = -K.mean(y_true * y_pred, axis=-1)
    return out #+ out2 # + K.std((y_true * y_pred)*alpha, axis=-1)
    #return out

def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    #remove all neutral predictions from y_pred and y_true
    #y_true = y_true[y_pred[:,1]!=True]
    #y_pred = y_pred[y_pred[:,1]!=True]    

    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #cce = 
    #cce = T.nnet.kullback_leibler_divergence(y_pred, y_true)
    #cce = -T.dot(y_pred, y_true)
    
    return cce


 
class PredictAheadModel (TradingModel):
    
    def plotPredictionsByNo (self, series_no, data='cv'):
        self.loadSeriesByNo(series_no)
        if data=='cv':
            my_pred = self.model.predict(self.dataset.cv_X)
        elif data=='test':
            my_pred = self.model.predict(self.dataset.test_X)
        fig = plt.figure ()
        plt.plot(my_pred, label="Return 7 days ahead")
        #plt.plot(my_pred[:,1], label="Neutral")
        #plt.plot(my_pred[:,2], label="Long")
        plt.legend(loc='best')
        plt.show ()
     
    def buildModel (self):
        # build the model: 2 stacked LSTM
        self.model = Sequential()
        self.model.add(LSTM(512, dropout_W=0.1, dropout_U=0.1, stateful=False, return_sequences=True, input_shape=(self.dataset.lookback_window, self.dataset.n_features)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(512, dropout_W=0.1, dropout_U=0.1, return_sequences=False))
        self.model.add(Dropout(0.2))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
        #self.model.add(Dropout(0.2))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(512))
        #self.model.add(Activation('sigmoid'))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(512))
        self.model.add(Dense(512, W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.0001)))
        self.model.add(Activation('softsign'))
        
        #self.model.add(Dropout(0.2))
        self.model.add(Dense(512, W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.0001)))
        self.model.add(Activation('softsign'))
        #self.model.add(Dropout(0.2))
        
        self.model.add(Dense(1, W_regularizer=l2(0.0001)))
        #self.model.add(Dense(1, activity_regularizer=activity_l2(0.01)))
        self.model.add(Activation('softsign'))
        #self.model.add(Activation('softmax'))
        self.model.compile(loss=_loss_tensor, 
            optimizer=SGD(lr=0.003))
        
        return self.model
        
my_train = PredictAheadModel (modelname=modelname,
                            modelpath=modelpath, n_features=142, 
                            lookback_window=126, cv_set_size=1500,
                            test_set_size=1000)
#my_train.dataset.n_features = 142
#my_train.lookback_window = 126
#my_train.batch_size = 128
#my_train.dataset.cv_set_size = 1500
#my_train.dataset.test_set_size = 1000
my_train.buildModel()
try:
    my_train.modelname = 'From_scratch_1'
    my_train.loadModel()
except:
    print ('Failed to load model')
    pass

my_train.dataset.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume'

cv_X = []
cv_y = []



for i in range (1, 5):
    my_train.loadSeriesByNo(5)
    my_train.modelname = 'From_scratch_'+str(i)+'_v2'
    
    pred = my_train.model.predict(my_train.dataset.cv_X)
    plt.figure()
    plt.plot(pred, label='CV Predictions')
    plt.legend ()
    plt.show ()
    
    #my_train.train ()
    
    #pred = my_train.model.predict(my_train.dataset.cv_X)
    #plt.plot(pred)
    
    ret = np.zeros (len(pred))
    
    for i in range (len(ret)):
        ret[i] = pred[i] * my_train.dataset.cv_y[i]
    plt.figure ()
    plt.plot(ret, label='Return')
    plt.legend ()
    plt.show ()
    
    plt.figure()
    plt.plot(np.cumsum(ret), label='cumulative return')
    plt.legend ()
    plt.show ()

if False:
    fig = plt.figure ()
    plt.plot(pred)
    plt.plot(my_train.dataset.cv_y)
    plt.show ()    
while False:
    for i in range (1, 18):
        #for i in range(40,41,1):
        #for i in [17,40]:
        
        my_train.modelname = 'From_scratch_'+str(i)+'_v2'
        try:
            my_train.loadModel ()    
        except:
            pass
        try:
            
            #17 - USDZAr
            #40 - USDBRL
            #41 - EURZAR?
            
            if True:
                #my_train.modelname = 'ZAR-late-sparsity'
                #my_train.dataset.loadDataSet(begin=i, end=i+5)
                #my_train.createSingleTrainSet()
                #my_train.loadSeriesByNo(i)
                
                
                #my_train.dataset.cv_X = my_train.dataset.test_X [0:1500,:,:]
                #my_train.dataset.cv_y = my_train.dataset.test_y [0:1500,:]
                
                #if i==0:
                #    cv_X = my_train.dataset.cv_X
                #    cv_y = my_train.dataset.cv_y
                #else:
                #    my_train.dataset.cv_X = cv_X
                #    my_train.dataset.cv_y = cv_y
                
                for j in range (10):
                        try:                
                            my_train.loadSeriesByNo(np.int(np.random.uniform(low=1, high=17)))
                        except:
                            raise ValueError('Did not manage to load series')
                        
                        my_train.dataset.X = my_train.dataset.X[0:2800,:,:]
                        my_train.dataset.y = my_train.dataset.y[0:2800,:]
                        
                        
                        my_train.train(shuffle=True)
                    
                my_train.modelname = 'From_scratch_'+str(i)+'_v2'
                my_train.model.save_weights(modelpath+'/'+my_train.modelname+".hdf5",overwrite=True)
                
                pred_train = my_train.model.predict(my_train.dataset.X)
                pred_cv = my_train.model.predict(my_train.dataset.cv_X)
                
                plt.figure()
                plt.plot(pred_train, label='pred train: '+str(i))
                plt.legend()
                #plt.plot(pred_cv)
                plt.show ()
                
                plt.figure()
                plt.plot(pred_cv, label='pred cv: '+str(i))
                plt.legend()
                #plt.plot(pred_cv)
                plt.show ()
                
                ret_train = np.zeros (len(pred_train))
                ret_train = pred_train * my_train.dataset.y
                ret_cv = np.zeros (len(pred_cv))
                ret_cv = pred_cv * my_train.dataset.cv_y
                #for i in range (len(ret)):
                #    if np.abs (pred[i]) > 0.2:
                #    ret[i] = np.max(-0.03, np.min(pred[i] * my_train.dataset.cv_y[i],0.05))
                #plt.plot(ret_train)
                
                plt.figure()
                plt.plot(np.cumsum(ret_cv), label='cv return')
                plt.legend()
                plt.show ()
                
                plt.figure ()
                plt.plot(np.cumsum(ret_train), label='train_return')
                plt.legend()
                plt.show ()
    
                
        except:
            print('Error')
            pass