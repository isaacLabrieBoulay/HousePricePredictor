from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
import numpy as np

def get_model():
    model = Sequential()
    # input is two features and we have 4 nodes in the first layer, using the tanh activation
    model.add(Dense(64, input_shape=(8,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    
    model.add(Dense(1))
    
    return model

def dnn(x_train, x_test, y_train):
    model = get_model()
    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=["mse", r_square])

    callbacks = [
    ReduceLROnPlateau(monitor='val_mse',
                  factor=0.4, 
                  patience=7, 
                  min_lr=0.000001, 
                  verbose=1),

    # add early stopping callback to prevent from overfitting
    EarlyStopping(monitor="val_mse", patience=15, mode='min', restore_best_weights=True)]
    history = model.fit(x_train, y_train, epochs=200, batch_size = 64,
                        shuffle=True, validation_split=0.2, callbacks = callbacks)

    y_pred = model.predict(x_test)
    return y_pred

def dnn_model(x_train, y_train):
    model = get_model()
    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=["mse", r_square])

    callbacks = [
    ReduceLROnPlateau(monitor='val_mse',
                  factor=0.4, 
                  patience=7, 
                  min_lr=0.000001, 
                  verbose=1),

    # add early stopping callback to prevent from overfitting
    EarlyStopping(monitor="val_mse", patience=15, mode='min', restore_best_weights=True)]
    history = model.fit(x_train, y_train, epochs=200, batch_size = 64,
                        shuffle=True, validation_split=0.2, callbacks = callbacks)

    # y_pred = model.predict(x_test)
    return model

def randomForest(x_train, x_test, y_train, n_est, max_depth, bootstrap):
    regressor = RandomForestRegressor(n_estimators = n_est, random_state = 0, max_depth=max_depth, bootstrap=bootstrap)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    return y_pred

def randomForest_model(x_train, x_test, y_train, n_est, max_depth, bootstrap):
    regressor = RandomForestRegressor(n_estimators = n_est, random_state = 0, max_depth=max_depth, bootstrap=bootstrap)
    regressor.fit(x_train, y_train)
    # y_pred = regressor.predict(x_test)
    return regressor

def knnr(x_train, x_test, y_train):
    neigh = KNeighborsRegressor(n_neighbors=5, metric="manhattan", weights="distance")
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)
    return y_pred

def knnr_model(x_train, y_train):
    neigh = KNeighborsRegressor(n_neighbors=5, weights="distance")
    neigh.fit(x_train, y_train)
    # y_pred = neigh.predict(x_test)
    return neigh

def gradientBoosting(x_train, x_test, y_train, learning_rate, n_estimators, max_depth):
    gbr = GradientBoostingRegressor(random_state=0, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    gbr.fit(x_train, y_train)
    y_pred = gbr.predict(x_test)
    
    return y_pred

def gradientBoosting_model(x_train, x_test, y_train, learning_rate, n_estimators, max_depth):
    gbr = GradientBoostingRegressor(random_state=0, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    #gbr = GradientBoostingRegressor(random_state=0, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    
    gbr.fit(x_train, y_train)
    # y_pred = gbr.predict(x_test)#
    
    return gbr

#Multi-Layer Perceptron
def mlp_regressor(x_train, x_test, y_train):
    mlp = MLPRegressor(random_state=0, max_iter=750)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    return y_pred

def svm_r(x_train, x_test, y_train):
    regr = svm.SVR()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    return y_pred
    
def kernel_ridge(x_train, x_test, y_train):
    krr = KernelRidge(alpha=1.0)
    krr.fit(x_train, y_train)
    y_pred = krr.predict(x_test)
    return y_pred

# Stochastic gradient descent regressor
def sgd_r(x_train, x_test, y_train):
    regr = SGDRegressor(max_iter=1000, tol=1e-3)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    return y_pred

def gaussianProcessRegressor(x_train, x_test, y_train):
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(x_train, y_train)
    y_pred = gpr.predict(x_test)
    return y_pred
    
def pls_r(x_train, x_test, y_train):
    pls2 = PLSRegression(n_components=2)
    pls2.fit(x_train, y_train)
    y_pred = pls2.predict(x_test)
    return y_pred

def decisionTree_reg(x_train, x_test, y_train):
    regr = DecisionTreeRegressor(random_state=0)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    return y_pred

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))
