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


def __traintest(X_train, X_test, y_train, y_test):

    #instance KNN model
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski',
                               leaf_size = 30, weights = 'uniform')
    knn.fit(X_train, y_train)
    print knn.score(X_train, y_train)
    print knn.score(X_test, y_test)

    return knn

def __loadsorttest(datastore = 'training_data/cleaned_data', test = False):

    #load in sorted SP data
    X_data, y_targets = loadtraindata(datastore)
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
    # add by KZI to plot 3D
    #plot3d(X_data, y_targets)
    ##########################
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_targets, test_size = 0.3, random_state = 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train) 
    X_test_std = sc.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter=2000)
    #mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100], max_iter=2000, activation='logistic')
    #mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30,30), max_iter=100, activation='relu')
    mlp.fit(X_train_std[:,[0,2,3,4,5,6]], y_train)
    print mlp
    predictions = mlp.predict(X_test_std[:,[0,2,3,4,5,6]])

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    #print X_train_std.shape
    #print X_train_std
    #print X_train_std[:,[0,2,4]]
    #model = __traintest(X_train_std[:,[0,2,3,4,5,6]],
    #                    X_test_std[:,[0,2,3,4,5,6]], y_train, y_test)
    
    return mlp, sc

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
            UnoccupiedY.append(X_data[i][1])
            UnoccupiedZ.append(X_data[i][2])
            i+=1
        if point == 1:
            SecondaryX.append(X_data[i][0])
            SecondaryY.append(X_data[i][1])
            SecondaryZ.append(X_data[i][2])
            i+=1
        if point == 2:
            PrimaryX.append(X_data[i][0])
            PrimaryY.append(X_data[i][1])
            PrimaryZ.append(X_data[i][2])
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
    __loadsorttest_nn()
    
"""
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
"""
