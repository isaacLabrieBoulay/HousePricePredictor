import algoTests
import dataProcessing
import algos
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

numpyArr = dataProcessing.prepData("training_set.csv")
scaledData, scaler = dataProcessing.scaleData(numpyArr)

def runAndTestRF():
    n_est = [1, 5, 10, 100, 1000]
    max_depths2 = [2, 3, 5, 8, 10]
    bootstrap = [True, False]

    maxr2RF = 0.8463203954661097
    ideal_n_est = 100
    ideal_depth_ = 10
    ideal_bootstrap = True
    mse = 0

    for n in n_est:
        for depth in max_depths2:
            for bool in bootstrap:
                r2 = algoTests.randomForest_test(scaledData, 10, n, depth, bool)
                if r2 > maxr2RF:
                    maxr2RF = r2
                    ideal_n_est = n
                    ideal_depth_ = depth
                    ideal_bootstrap = bool
                    print("Max r2:")
                    print(r2)
                    print("ideal estimators:")
                    print(n)
                    print("ideal depth")
                    print(depth)
                    print("ideal bootstrap:")
                    print(bool)
                    print("")
    
def runAndTestGB():
    learning_rates = [0.000625, 0.00125, 0.0025, 0.005] 
    n_estimatorsArr = [2500, 5000, 10000, 18500] 
    max_depths = [3, 6, 9]

    ideal_learningRate = 0.000625
    ideal_n_estimator = 18500
    ideal_depth = 3
    mse = 0

    maxr2 = 0.8549370158946126

    for lr in learning_rates:
        for n in n_estimatorsArr:
            for depth in max_depths:
                r2 = algoTests.gradientBoosting_test(scaledData, 10, lr, n, depth)
                if r2 > maxr2:
                    print("new r2: ")
                    print(r2)
                    print("new ideal lr: ")
                    print(lr)
                    print("new ideal n est: ")
                    print(n)
                    print("New ideal depth")
                    print(depth)
                    ideal_learningRate = lr 
                    ideal_n_estimator = n
                    ideal_depth = depth
                    maxr2 = r2
                    print("")


    print("Best r2 score: ")
    print(maxr2)
    print("")
    print("Ideal learning rate: ")
    print(ideal_learningRate)
    print("")
    print("Ideal number of estimators: ")
    print(ideal_n_estimator)
    print("")
    print("Ideal depth:")
    print(ideal_depth)
    print("")

def runOptimalRF():
    r2 = algoTests.randomForest_test(scaledData, 10, 100, 10, True)
    
def runOptimalGB():
    r2 = algoTests.gradientBoosting_test(scaledData, 10, 0.000625, 18500, 3)

def runOptimalKNN():
    r2 = algoTests.knnr_test(scaledData, 10)
    
def runOptimalDNN():
    r2 = algoTests.dnn_test(scaledData, 10)

def saveOptimal_RFModel():
    algoTests.randomForest_getBestModel(scaledData, 30)
    
def saveOptimal_GBModel():
    algoTests.gradientBoost_getBestModel(scaledData, 30)
    
def saveOptimal_KNNModel():
    algoTests.knn_getBestModel(scaledData, 30)

def saveOptimal_DNNModel():
    algoTests.dnn_getBestModel(scaledData, 30)


algoTests.individual_price_predictor_geo()