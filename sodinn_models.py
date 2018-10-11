"""
Discriminative models: convLSTM2d, conv3d, RF.
"""
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['train_random_forest',
           'train_convlstm2d',
           'train_conv3d',
           'make_parallel']


from vip_hci.conf import time_ini, timing, time_fin
from munch import *
#from keras.backend import set_floatx
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
#from keras.layers.advanced_activations import PReLU
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
#from keras.regularizers import l2, l1, l1_l2
from keras.utils.np_utils import to_categorical
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf

#from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
#from keras.layers.normalization import BatchNormalization

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             accuracy_score)
from .sodinn_utils import save_res



def train_random_forest(X, Y, test_size=0.1, n_estimators=100, criterion='gini', 
                        max_depth=None, min_samples_split=2, 
                        min_samples_leaf=1, min_weight_fracleaf=0.0, 
                        max_features='auto', max_leaf_nodes=None, 
                        min_impurity_split=1e-07, bootstrap=True, 
                        oob_score=True, n_jobs=50, random_state=None, 
                        verbose=0, class_weight=None, save=None):
    """
    oob_score : bool (default=True)
        Whether to use out-of-bag samples to estimate the generalization accuracy.


    warm_start=False

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    starttime = time_ini()

    # Vectorizing the MLAR samples
    n_samples = X.shape[0]
    Xbig_flat = X.reshape(n_samples, -1)

    ### Mixed train/test sets with SKLEARN split
    X_train, X_test, y_train, y_test = train_test_split(Xbig_flat, Y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    # Instantiating the random forest classfier 
    rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, 
                                min_weight_fraction_leaf=min_weight_fracleaf, 
                                max_features=max_features, 
                                max_leaf_nodes=max_leaf_nodes, 
                                min_impurity_split=min_impurity_split, 
                                bootstrap=bootstrap, oob_score=oob_score, 
                                n_jobs=n_jobs, random_state=random_state, 
                                verbose=verbose, class_weight=class_weight)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Score of the training dataset obtained using an out-of-bag estimate.
    if oob_score: print("Out-of-bag score: {}".format(rf.oob_score_))
    print("Accuracy on test set: {}".format(accuracy_score(y_test, y_pred)))
    print("Confusion matrix (true X pred)")
    print(confusion_matrix(y_test, y_pred))

    timing(starttime)
    fintime = time_fin(starttime)

    if save is not None:
        save_res(save + '.p', rf)

    return rf



def train_convlstm2d(X, Y, test_size=0.1, validation_split=0.1,
                     random_state=None, ksize1=(3,3), filters1=40,
                     poolsi1=(2,2,2), strides1=(1,1), ksize2=(3,3), 
                     filters2=40, poolsi2=(2,2,2), strides2=(1,1),
                     dense_units=128, activation='relu', learnrate=0.003,
                     batchsize=512, epochs=20, patience=5, min_delta=0.01, 
                     output_activation='sigmoid', retrain=None, verb=1,
                     summary=True, ngpus=1, save=None):
    """
 
    """
    starttime = time_ini()

    ### Mixed train/test sets with SKLEARN split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=test_size, 
                                                    random_state=random_state)
    msg = 'Zeros in train: {} |  Ones in train: {}'
    print(msg.format(y_train.tolist().count(0), y_train.tolist().count(1)))
    msg = 'Zeros in test: {} |  Ones in test {}:'
    print(msg.format(y_test.tolist().count(0), y_test.tolist().count(1)))

    if output_activation=='softmax':
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    ### adding the "channels" dimension (1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 
                              X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 
                            X_test.shape[3], 1)

    print("\nShapes of train and test sets before CNN:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print()

    if retrain is not None:
        M = retrain
        ### Training the network
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                       min_delta=min_delta, verbose=verb)

        hist = M.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, 
                     initial_epoch=0, verbose=verb, validation_split=0.1, 
                     callbacks=[early_stopping], shuffle=True)

        score = M.evaluate(X_test, y_test, verbose=verb)
        print('\n Test score/loss:', score[0])
        print(' Test accuracy:', score[1])

        fintime = time_fin(starttime)
        bunch_results = Munch(hist=hist.history, runtime=fintime, score=score,
                              trainshape=X_train.shape, testshape=X_test.shape,
                              testsize=test_size, valsplit=validation_split,
                              kernsize1=ksize1, filters1=filters1, 
                              poolsize1=poolsi1, kernsize2=ksize2, 
                              filters2=filters2, poolsize2=poolsi2, 
                              denunits=dense_units, activation=activation,
                              learnrate=learnrate, batchsize=batchsize, 
                              ngpus=ngpus, maxepochs=epochs, patience=patience, 
                              mindelta=min_delta)
        
        if save is not None and isinstance(save, str):
            save_res(save+'.p', bunch_results)
            M.save(save+'.h5')
            timing(starttime)
        
        return M

    ############################################################################
    ### Creating the NN model
    M = Sequential()
    n_pcs = X_train.shape[1]
    patch_size = X_train.shape[2]

    ### First convLSTM2D layer
    M.add(ConvLSTM2D(filters=filters1, kernel_size=ksize1, strides=strides1,
                     input_shape=(n_pcs, patch_size, patch_size, 1),  #n_pcs can be None
                     padding='same', return_sequences=True,
                     dilation_rate=(1, 1), activation='tanh', 
                     recurrent_activation='hard_sigmoid', use_bias=True, 
                     kernel_initializer='glorot_uniform', 
                     recurrent_initializer='orthogonal', 
                     bias_initializer='zeros', unit_forget_bias=True, 
                     kernel_regularizer=None, recurrent_regularizer=None, 
                     bias_regularizer=None, activity_regularizer=None, 
                     kernel_constraint=None, recurrent_constraint=None, 
                     bias_constraint=None, go_backwards=False, stateful=False, 
                     dropout=0., recurrent_dropout=0.))
    #M.add(BatchNormalization())
    M.add(MaxPooling3D(pool_size=poolsi1, strides=(2, 2, 2),
                       border_mode='valid', name='pooling_layer1'))
    
    ### Second convLSTM2D layer
    M.add(ConvLSTM2D(filters=filters2, kernel_size=ksize2, strides=strides2,
                     padding='same', return_sequences=True,
                     dilation_rate=(1, 1), activation='tanh', 
                     recurrent_activation='hard_sigmoid', use_bias=True, 
                     kernel_initializer='glorot_uniform', 
                     recurrent_initializer='orthogonal', 
                     bias_initializer='zeros', unit_forget_bias=True, 
                     kernel_regularizer=None, recurrent_regularizer=None, 
                     bias_regularizer=None, activity_regularizer=None, 
                     kernel_constraint=None, recurrent_constraint=None, 
                     bias_constraint=None, go_backwards=False, stateful=False, 
                     dropout=0., recurrent_dropout=0.))
    #M.add(BatchNormalization())
    M.add(MaxPooling3D(pool_size=poolsi2, strides=(2, 2, 2),
                       border_mode='valid', name='pooling_layer2'))

    M.add(Flatten(name='flatten'))

    # Fully-connected layer with ``units`` hidden units
    M.add(Dense(units=dense_units, name='dense_128units'))
    M.add(Activation(activation, name='activ_dense'))
    #M.add(BatchNormalization())
    M.add(Dropout(rate=0.5, name='dropout_dense'))

    if output_activation=='sigmoid':
        M.add(Dense(units=1, name='dense_1units'))
    else:
        M.add(Dense(units=2, name='dense_2units'))

    M.add(Activation(output_activation, name='activ_out'))

    if summary: M.summary()

    ### Multi-GPUs
    if ngpus > 1: M = make_parallel(M, ngpus)
    ### Training the network
    M.compile(loss='binary_crossentropy', metrics=['accuracy'],
              optimizer=Adam(lr=learnrate, decay=1e-2))
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                   min_delta=min_delta, verbose=verb)

    hist = M.fit(X_train, y_train, batch_size=batchsize*ngpus, epochs=epochs, 
                initial_epoch=0, verbose=verb, validation_split=validation_split, 
                callbacks=[early_stopping], shuffle=True)

    score = M.evaluate(X_test, y_test, verbose=verb)
    print('\n Test score/loss:', score[0])
    print(' Test accuracy:', score[1])

    timing(starttime)
    fintime = time_fin(starttime)

    #return M, hist
    bunch_results = Munch(hist=hist.history, runtime=fintime, score=score,
                          trainshape=X_train.shape, testshape=X_test.shape,
                          testsize=test_size, valsplit=validation_split,
                          kernsize1=ksize1, filters1=filters1, poolsize1=poolsi1, 
                          kernsize2=ksize2, filters2=filters2, poolsize2=poolsi2, 
                          denunits=dense_units, activation=activation,
                          learnrate=learnrate, batchsize=batchsize, ngpus=ngpus,
                          maxepochs=epochs, patience=patience, mindelta=min_delta)
    
    if save is not None:
        save_res(save+'.p', bunch_results)
        M.save(save+'.h5')
        timing(starttime)
    
    return M



def train_conv3d(X, Y, test_size=0.1, validation_split=0.1, random_state=0,
                  ksize1=(4,3,3), filters1=16, poolsi1=(2,2,2), strides1=(1,1,1),
                  ksize2=(4,3,3), filters2=64, poolsi2=(2,2,2), strides2=(1,1,1),
                  dense_units=128, activation='relu', learnrate=0.003, 
                  batchsize=512, epochs=20, patience=2, min_delta=0.01,
                  retrain=None, verb=1, summary=True, ngpus=1, save=None):
    """
    kernel sizes tested:
    (3,3,3) (10,3,3) (10,9,9)
    (3,3,3) (5,3,3) (5,9,9)


    """
    starttime = time_ini()

    ### Mixed train/test sets with SKLEARN split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=test_size, 
                                                    random_state=random_state)
    msg = 'Zeros in train: {} |  Ones in train:'
    print(msg.format(y_train.tolist().count(0), y_train.tolist().count(1)))
    msg = 'Zeros in test: {} |  Ones in test:'
    print(msg.format(y_test.tolist().count(0), y_test.tolist().count(1)))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    ### adding the "channels" dimension (1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 
                              X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 
                            X_test.shape[3], 1)

    print("\nShapes of train and test sets before CNN:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print()

    if retrain is not None:
        M = retrain
        ### Training the network
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                       min_delta=min_delta, verbose=verb)

        hist = M.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, 
                     initial_epoch=0, verbose=verb, validation_split=0.1, 
                     callbacks=[early_stopping], shuffle=True)

        score = M.evaluate(X_test, y_test, verbose=verb)
        print('\n Test score/loss:', score[0])
        print(' Test accuracy:', score[1])

        return M, hist

    ############################################################################
    ### Creating the NN model
    M = Sequential()
    n_pcs = X_train.shape[1]
    patch_size = X_train.shape[2]

    ### 1st layer group
    M.add(Convolution3D(filters=filters1, kernel_size=ksize1, strides=strides1,
                         padding='same', kernel_initializer='glorot_uniform',
                         bias_initializer='random_normal', #'zeros'
                         input_shape=(n_pcs, patch_size, patch_size, 1), 
                         name='conv_layer1'))
    M.add(Activation(activation, name='activ_layer1'))
    #M.add(BatchNormalization())   # maintains the mean activation ~ 0 and stddev ~ 1
    M.add(MaxPooling3D(pool_size=poolsi1, strides=(2, 2, 2),
                       border_mode='valid', name='pooling_layer1'))

    ### 2nd layer group
    M.add(Convolution3D(filters=filters2, kernel_size=ksize2, padding='same',
                        strides=strides2, kernel_initializer='glorot_uniform', 
                        name='conv_layer2'))
    M.add(Activation(activation, name='activ_layer2'))
    #M.add(BatchNormalization())
    M.add(MaxPooling3D(pool_size=poolsi2, strides=(2, 2, 2),
                       border_mode='valid', name='pooling_layer2'))
    M.add(Dropout(rate=0.25, name='dropout_layer2'))

    M.add(Flatten(name='flatten'))

    # Dense(128) is a fully-connected layer with 128 hidden units
    M.add(Dense(units=dense_units, name='dense_128units'))
    M.add(Activation(activation, name='activ_dense'))
    #M.add(BatchNormalization())
    M.add(Dropout(rate=0.5, name='dropout_dense'))

    M.add(Dense(units=2, name='dense_2units'))
    M.add(Activation('softmax', name='activ_softmax'))

    if summary: M.summary()

    ### Multi-GPUs
    if ngpus > 1: M = make_parallel(M, ngpus)
    ### Training the network
    M.compile(loss='binary_crossentropy', metrics=['accuracy'],
              optimizer=Adam(lr=learnrate, decay=1e-2))
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                   min_delta=min_delta, verbose=verb)

    hist = M.fit(X_train, y_train, batch_size=batchsize*ngpus, epochs=epochs, 
                initial_epoch=0, verbose=verb, validation_split=validation_split, 
                callbacks=[early_stopping], shuffle=True)

    score = M.evaluate(X_test, y_test, verbose=verb)
    print('\n Test score/loss:', score[0])
    print(' Test accuracy:', score[1])

    timing(starttime)
    fintime = time_fin(starttime)

    #return M, hist
    bunch_results = Munch(hist=hist.history, runtime=fintime, score=score,
                          trainshape=X_train.shape, testshape=X_test.shape,
                          testsize=test_size, valsplit=validation_split,
                          kernsize1=ksize1, filters1=filters1, poolsize1=poolsi1, 
                          kernsize2=ksize2, filters2=filters2, poolsize2=poolsi2, 
                          denunits=dense_units, activation=activation,
                          learnrate=learnrate, batchsize=batchsize, ngpus=ngpus,
                          maxepochs=epochs, patience=patience, mindelta=min_delta)
    
    if save is not None:
        save_res('Model_3dcnn_'+save+'.p', bunch_results)
        M.save('Model_3dcnn_'+save+'.h5')
        timing(starttime)
    
    return M



def make_parallel(model, gpu_count):
    """
    https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
    """
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, 
                                     arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)
