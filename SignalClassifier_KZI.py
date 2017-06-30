#gr-spacebase project github.com/wirrell/gr-spacebase
#This file contains the training for the signal prediction model

import numpy as np
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from DataCollect import loadtraindata

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

import numpy as np
import tensorflow as tf
import time
import random
import cPickle
import struct
import os
import keras

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


from keras.optimizers  import Adam
from keras.constraints import MaxNorm

import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam


from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from tensorflow.contrib.session_bundle import exporter
from sklearn.svm import SVC


def __traintest(X_train, X_test, y_train, y_test):

    #instance KNN model
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski',
                               leaf_size = 30, weights = 'uniform')
    knn.fit(X_train, y_train)
    print knn.score(X_train, y_train)
    print knn.score(X_test, y_test)

    return knn


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s','x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1,0], y=X[y == c1,1],
            alpha=0.8, c=cmap(idx), 
            marker=markers[idx], label=c1)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o',
            s=55, label='test set')


## \brief Loads RadioML data
def load_TVWS_Data(datastore = 'training_data/cleaned_data'):
    # Load the dataset ...
    '''
    #  You will need to seperately download or generate this file
    #Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
    #snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)

    # Partition the data
    #  into training and test sets of the form we can train/test on 
    #  while keeping SNR and Mod labels handy for each
    '''
    X_data, y_targets = loadtraindata(datastore)

    np.random.seed(2016)
    n_examples = X_data.shape[0]
    n_train = int(n_examples * 0.7)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    #print train_idx
    print "train_idx: size, type:", len(train_idx), type(train_idx)
    #print test_idx
    print "test_idx size, type", len(test_idx), type(test_idx)
    #X_train = X_data[train_idx]  # all 7 features used
    X_train_tmp = X_data[train_idx]
    X_train = X_train_tmp[:,[0,2,3,4,5,6]]  # just 6 features are used
    #print X_train

    #X_test =  X_data[test_idx]
    X_test_tmp =  X_data[test_idx]
    X_test = X_test_tmp[:,[0,2,3,4,5,6]]
    #print X_test
    #print y_targets[train_idx]
    Y_train = to_onehot(map(lambda x: int(y_targets[x]), train_idx))
    Y_test = to_onehot(map(lambda x: int(y_targets[x]), test_idx))
    #print Y_train

    in_shp = list(X_train.shape[1:])

    print X_train.shape, in_shp
    print X_train

    #dr = 0.5
    # Create model
    model = models.Sequential()
    model.add(Dense(12, input_dim=6, activation='relu')) # six features
    #model.add(Dense(12, input_dim=7, activation='relu')) # seven features
    model.add(Dense(8, activation='relu'))
    #model.add(Dense(3,activation='sigmoid'))
    model.add(Dense(3,activation='softmax')) # ouput = [x,x,x]

    #Compil model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    #Fit the model
    # batch_size 30, epoch 40 is best score:0.87
    batch_size = 30
    epoch = 60
    filepath = 'TVWS_CNN.wts.h5'
    #model.fit(X_train, Y_train, nb_epoch=150, batch_size=10)
    #model.fit(X_train, Y_train, batch_size=1, nb_epoch=61, verbose=2, validation_data=(X_test, Y_test))
    history=model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch, verbose=2)
    #model.load_weights(filepath)
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    print score

    Y_predic = model.predict(X_test, batch_size=1)
    print Y_predic
    for i in range(0, Y_predic.shape[0]):
        k = int(np.argmax(Y_predic[i,:]))
        print k

    
    #print Y_predic
    #print Y_predic.shape
    #print Y_test

    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    #plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    #plt.show()

    test_Y = model.predict(X_test)

    


    return X_train,Y_train,X_test,Y_test,train_idx,test_idx


def __loadsorttest(datastore = 'training_data/cleaned_data', test = False):

    #load in sorted SP data
    X_data, y_targets = loadtraindata(datastore)
    print X_data.shape, type(y_targets)
    # add by KZI to plot 3D
    #plot3d(X_data, y_targets)
    ##########################
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_targets, test_size = 0.3, random_state = 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train) 
    X_test_std = sc.transform(X_test)

    print X_train_std.shape
    #print X_train_std
    #print X_train_std[:,[0,2,4]]
    model = __traintest(X_train_std[:,[0,2,3,4,5,6]],
                        X_test_std[:,[0,2,3,4,5,6]], y_train, y_test)
    
    return model, sc

def __loadsorttest_nn(datastore = 'training_data/cleaned_data', test = False):

    #load in sorted SP data
    X_data, y_targets = loadtraindata(datastore)
    print X_data.shape
    print y_targets
    # add by KZI to plot 3D
    plot3d(X_data, y_targets)
    ##########################
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_targets, test_size = 0.3, random_state = 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train) 
    X_test_std = sc.transform(X_test)

    #mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter=2000)
    mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100], max_iter=2000, activation='logistic')
    #mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30,30), max_iter=100, activation='relu')
    mlp.fit(X_train_std[:,[0,2,3,4,5,6]], y_train)
    #mlp.fit(X_train_std, y_train)
    print mlp
    predictions = mlp.predict(X_test_std[:,[0,2,3,4,5,6]])
    #predictions = mlp.predict(X_test_std)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    #print X_train_std.shape
    #print X_train_std
    #print X_train_std[:,[0,2,4]]
    #model = __traintest(X_train_std[:,[0,2,3,4,5,6]],
    #                    X_test_std[:,[0,2,3,4,5,6]], y_train, y_test)
    
    return mlp, sc

def __loadsorttest_svm(datastore = 'training_data/cleaned_data', test = False):

    #load in sorted SP data
    X_data, y_targets = loadtraindata(datastore)
    print X_data.shape
    print y_targets.shape
    # add by KZI to plot 3D
    #plot3d(X_data, y_targets)
    ##########################
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_targets, test_size = 0.3, random_state = 0)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train) 
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    svm = SVC(kernel='linear', C=1.0, random_state=0)
    #svm = SVC(kernel='rbf', C=1.0, random_state=0)
    #svm = SVC(kernel='rbf', C=0.5, random_state=0)
    #svm = SVC(kernel='rbf', C=1.0, random_state=0)
    svm.fit(X_train_std[:,[0,2,3,4,5,6]], y_train)

    predict = svm.predict(X_test_std[:,[0,2,3,4,5,6]])
    print 'Predict:',predict
    print 'y_test:',y_test
    print svm.score(X_test_std[:,[0,2,3,4,5,6]],y_test)


    '''
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    '''
    
    return svm, sc



def loadmodel(datastore):

    model, sc = __loadsorttest(datastore)
    
    return model, sc

def plot3d(X_data, y_targets):

    #plot data in 3 planes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    PrimaryX = []
    PrimaryY = []
    PrimaryZ = []
    SecondaryX = [] 
    SecondaryY = []
    SecondaryZ = []
    UnoccupiedX = []
    UnoccupiedY = []
    UnoccupiedZ = []

    
    Secondary = []
    Unoccupied = []
    for point in y_targets:
        if point == 0:
            UnoccupiedX.append(X_data[i][0])
            UnoccupiedY.append(X_data[i][3])   #1
            UnoccupiedZ.append(X_data[i][4])   #2
            i+=1
        if point == 1:
            SecondaryX.append(X_data[i][0])
            SecondaryY.append(X_data[i][3])    #1
            SecondaryZ.append(X_data[i][4])    #2
            i+=1
        if point == 2:
            PrimaryX.append(X_data[i][0])
            PrimaryY.append(X_data[i][3])      #1
            PrimaryZ.append(X_data[i][4])      #2
            i+=1
        if point == 4:
            #Used for unknown, currently obsoltete
            i+=1
    ax.scatter(UnoccupiedX, UnoccupiedY, UnoccupiedZ, c='g', marker
               ='o', label = 'Unoccupied')
    ax.scatter(SecondaryX, SecondaryY, SecondaryZ, c='b', marker
               ='^', label = 'Secondary')
    ax.scatter(PrimaryX, PrimaryY, PrimaryZ, c='r', marker
               ='x', label = 'Primary')
    ax.legend(bbox_to_anchor=(1.10,1.10))
    ax.autoscale(tight = True)
    ax.view_init(elev=25., azim=-125)
    ax.set_xlabel('Max power (dBm)')
    ax.set_ylabel('Relative mean power (dBm)')
    ax.set_zlabel('Std dev power (dBm)')
    
    plt.show()
    



if __name__ == '__main__':
    #__loadsorttest_nn()
    #model,scaler=__loadsorttest_svm()
    #model,scaler=__loadsorttest_nn()
    model,scaler = loadmodel('training_data/cleaned_data')

    print type(model), type(scaler)

    print 'Real testing:'
    X_data, y_targets = loadtraindata('training_data/test_data')
    print X_data.shape
    print y_targets.shape
        
    X_std = scaler.transform(X_data) 
    
    #X_combined_std = np.vstack(X_std)
    #y_combined = np.hstack(y_targets)

    predict = model.predict(X_std[:,[0,2,3,4,5,6]])
    print 'Predict:',predict
    print 'y_targets:',y_targets
    print model.score(X_std[:,[0,2,3,4,5,6]],y_targets)


    #X_train,Y_train,X_test,Y_test,train_idx,test_idx = load_TVWS_Data()
    
'''
    #model, sc = __loadsorttest(test=True)
    kzi, sc = __loadsorttest()
    XX_data, yy_targets = loadtraindata('training_data/test_data')
    #sc.fit(X_data)
    #X_data_std = sc.transform(X_data) 
    #print type(model)
    #print model
    #print XX_data[:,[0,2,3,4,5,6]]
    kzi = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski',
                               leaf_size = 30, weights = 'uniform')
    kzi.fit(XX_data[:,[0,2,3,4,5,6]], yy_targets)
   


    print kzi.predict(XX_data[:,[0,2,3,4,5,6]])
    print kzi.score(XX_data[:,[0,2,3,4,5,6]],yy_targets)
    print yy_targets
'''
