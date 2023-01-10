import algos
from sklearn.metrics import r2_score
from helpers import standard_deviation
from sklearn import preprocessing
import dataProcessing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
import tensorflow as tf
from geopy.geocoders import Nominatim
import math

def randomForest_getBestModel(scaledData, iterations):
    #Random Forest iterations
    bestR2 = 0
    bestModel = None
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        model = algos.randomForest_model(x_train, x_test, y_train, 100, 10, True)
        
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        
        if r2 > bestR2:
            bestR2 = r2
            bestModel = model
    
    if bestModel != None:
        pickle.dump(bestModel, open('best_RandomForest.sav', 'wb'))

def gradientBoost_getBestModel(scaledData, iterations):
    bestR2 = 0
    bestModel = None
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)
        
        model = algos.gradientBoosting_model(x_train, x_test, y_train, 0.000625, 18500, 3)
        y_pred = model.predict(x_test)
        
        r2 = r2_score(y_test, y_pred)
        
        if r2 > bestR2:
            bestR2 = r2
            bestModel = model
    
    if bestModel != None:
        pickle.dump(bestModel, open('best_gradientBoost.sav', 'wb'))
        

def knn_getBestModel(scaledData, iterations):
    bestR2 = 0
    bestModel = None
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split_KNN(scaledData)

        
        model = algos.knnr_model(x_train, y_train)
        y_pred = model.predict(x_test)
        
        r2 = r2_score(y_test, y_pred)
        print(r2)
        if r2 > bestR2:
            print("new best r2")
            bestR2 = r2
            bestModel = model
    
    if bestModel != None:
        pickle.dump(bestModel, open('best_knn.sav', 'wb'))
    
def dnn_getBestModel(scaledData, iterations):
    bestR2 = 0
    bestModel = None
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)
        
        model = algos.dnn_model(x_train, y_train)
        y_pred = model.predict(x_test)
        
        r2 = r2_score(y_test, y_pred)
        
        if r2 > bestR2:
            bestR2 = r2
            bestModel = model
    
    if bestModel != None:
        bestModel.save("best_dnn.h5")

def randomForest_test(scaledData, iterations, n_est, max_depth, bootstrap):
    
    #Random Forest iterations
    total_r2_Score = 0
    individual_r2_scores = []
    total_mse = 0
    individual_mse = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.randomForest(x_train, x_test, y_train, n_est, max_depth, bootstrap)
        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse
        individual_mse.append(mse)
        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
    #calculating the average R2
    average_mse = total_mse / iterations
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for RandomForest: ")
    print(average_r2)
    print("Standard Deviation of Random Forest r2:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2)) 
    print("Average MSE for Random Forest:")
    print(average_mse)
    print("Standard Deviation of MSE")
    print(standard_deviation(iterations, individual_mse, average_mse))
       
    
    return average_r2
    
def knnr_test(scaledData, iterations):
    
    total_r2_Score = 0
    individual_r2_scores = []
    total_mse = 0
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split_KNN(scaledData)

        y_pred = algos.knnr(x_train, x_test, y_train)
        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse
        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    average_mse = total_mse / iterations
    print("Average R2 score for K Nearest Neighbor regressor: ")
    print(average_r2)

    print("Standard Deviation of K Nearest Neighbor regressor:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    
    print("Average MSE: ")
    print(average_mse)

def gradientBoosting_test2(scaledData, scaler):
    x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)
    y_pred = algos.gradientBoosting(x_train, x_test, y_train)
    r2 = r2_score(y_test, y_pred)
    print(y_test.shape)
    print(y_pred.shape)
    reshaped_yPred = np.reshape(y_pred, (len(y_pred), 1))
    reshaped_yTest = np.reshape(y_test, (len(y_test), 1))
    
    lat = x_test[:,0]
    lng = x_test[:,1]
    price_pred = y_pred
    price_real = y_test
    bed = x_test[:,2]
    bath = x_test[:,3]
    square = x_test[:,4]
    year = x_test[:,5]
    residential = x_test[:,6]
    condo = x_test[:, 7]
    
    empty_array1 = np.empty((116, 0), float)
    xtest_plus_yTest = np.append(empty_array1, np.array([lat]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([lng]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([price_real]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([bed]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([bath]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([square]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([year]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([residential]).transpose(), axis=1)
    xtest_plus_yTest = np.append(xtest_plus_yTest, np.array([condo]).transpose(), axis=1)
    
    empty_array2 = np.empty((116, 0), float)
    xtest_plus_yPred = np.append(empty_array2, np.array([lat]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([lng]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([price_pred]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([bed]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([bath]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([square]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([year]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([residential]).transpose(), axis=1)
    xtest_plus_yPred = np.append(xtest_plus_yPred, np.array([condo]).transpose(), axis=1)
    
    print("EstimatedValues shape:")
    print(xtest_plus_yTest.shape)
    print("Predicted values shape:")
    print(xtest_plus_yPred.shape)
    
    estimatedTestValues = scaler.inverse_transform(xtest_plus_yPred)
    realTestValues = scaler.inverse_transform(xtest_plus_yTest)
    
    predicatedVals_df = pd.DataFrame(estimatedTestValues, columns = ['latitude','longitude', 'predicted_price', 'bed', 'bath', 'square', 'year', 'residential', 'condo'])
    realVals_df = pd.DataFrame(realTestValues, columns = ['latitude','longitude', 'real_price', 'bed', 'bath', 'square', 'year', 'residential', 'condo'])
    
    print("predicatedVals: ")
    print(predicatedVals_df)
    print("")
    print("")
    print("")
    print("realVals: ")
    print(realVals_df)
    print("")
    print("")

def gradientBoosting_test(scaledData, iterations, learning_rate, n_estimators, max_depth):
    total_r2_Score = 0
    individual_r2_scores = []
    total_mse = 0
    individual_mse = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.gradientBoosting(x_train, x_test, y_train, learning_rate, n_estimators, max_depth)
        
        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse
        individual_mse.append(mse)
        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
    #calculating the average R2
    average_mse = total_mse / iterations
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for Gradient Boosting: ")
    print(average_r2)
    print("Standard Deviation of Gradient Boosting:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    print("Average MSE for Gradient Boost:")
    print(average_mse)
    print("Standard Deviaton of mse:")
    print(standard_deviation(iterations, individual_mse, average_mse))
    return average_r2
    
    
def dnn_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    total_mse = 0
    for x in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)
        y_pred = algos.dnn(x_train, x_test, y_train)
        
        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse
        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for DNN: ")
    print(average_r2)

    print("Standard Deviation of R2 DNN:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    
    print("Avergae MSE: ")
    print(total_mse / iterations)
    return average_r2
    
    
        
    
def mlp_regressor_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.mlp_regressor(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for MLP: ")
    print(average_r2)

    print("Standard Deviation of MLP:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))


def svm_r_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.svm_r(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for SVM-R:")
    print(average_r2)

    print("Standard Deviation of SVM-R:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    
def kernel_ridge_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.kernel_ridge(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for Kernel-ridge:")
    print(average_r2)

    print("Standard Deviation of kernel-ridge:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    
def sgd_r_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.sgd_r(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for sgd_r:")
    print(average_r2)

    print("Standard Deviation of sgd_r:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    
    
def gaussianProcessRegressor_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.gaussianProcessRegressor(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for gaussianProcessRegressor:")
    print(average_r2)

    print("Standard Deviation of gaussianProcessRegressor:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))

def pls_r_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.pls_r(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the avereshapedrage R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for pls_r:")
    print(average_r2)

    print("Standard Deviation of pls_r:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))
    
def decisionTree_reg_test(scaledData, iterations):
    total_r2_Score = 0
    individual_r2_scores = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = dataProcessing.split(scaledData)

        y_pred = algos.decisionTree_reg(x_train, x_test, y_train)

        r2 = r2_score(y_test, y_pred)
        total_r2_Score += r2
        individual_r2_scores.append(r2)
    
     #calculating the average R2
    average_r2 = total_r2_Score / iterations
    print("Average R2 score for decisionTree_reg:")
    print(average_r2)

    print("Standard Deviation of decisionTree_reg:")
    print(standard_deviation(iterations, individual_r2_scores, average_r2))


def price_predictor():

    method = "AVG"
    user_estimator = input("Which estimator would you like to use? (RF, GB, KNN, DNN, or AVG) ")

    if(user_estimator != ""):
        # Print welcome message if the value is not empty
        if user_estimator == "RF":
            method = user_estimator
        elif user_estimator == "GB":
            method = user_estimator
        elif user_estimator == "KNN":
            method = user_estimator
        elif user_estimator == "DNN":
            method = user_estimator
        elif user_estimator == "AVG":
            method = user_estimator
        else:
            print("invalid estimator therefore defaulting to AVG")
        
    else:
        # Print empty message
        print("Please pick.")
        exit()

    # let user know which estimator they chose
    print("The " + str(method) + " estimator will be used.")

    # get the test data
    data = dataProcessing.prepData("../test_set.csv")

    # scale the data
    scaled_data, scaler = dataProcessing.scaleData(data)

    # Get y and X
    y = scaled_data[:,2]
    #print("shape of y: " + str(y.shape))
    # initialize prediction array
    y_pred = np.empty((len(y),))
    #print("shape of y_pred: " + str(y_pred.shape))


    X = np.delete(scaled_data, 2, axis=1)
    # Get X data for KNN (lat., long., sqrft, num. bathrooms, and condo)
    X_KNN = np.delete(scaled_data, [2,3,6,7], axis=1)



    if method == "RF":
        
        # load the model
        model = pickle.load(open("best_RandomForest.sav", 'rb'))
        # make the prediction
        y_pred = model.predict(X)


    elif method == "GB":
        
        # load the model
        model = pickle.load(open("best_gradientBoost.sav", 'rb'))
        # make the prediction
        y_pred = model.predict(X)


    elif method == "KNN":

        # load the model
        model = pickle.load(open("best_knn.sav", 'rb'))
        # make the prediction
        y_pred = model.predict(X_KNN)


    elif method == "DNN":

        # load model
        model = tf.keras.models.load_model("best_dnn.h5", compile=False)
        # make the prediction
        y_pred = model.predict(X)


    else:

        # load all models
        RF = pickle.load(open("best_RandomForest.sav", 'rb'))
        GB = pickle.load(open("best_gradientBoost.sav", 'rb'))
        KNN = pickle.load(open("best_knn.sav", 'rb'))
        DNN = tf.keras.models.load_model("best_dnn.h5", compile=False)

        # make all predictions
        y_pred_RF = RF.predict(X)
        y_pred_GB = GB.predict(X)
        y_pred_KNN = KNN.predict(X_KNN)
        y_pred_DNN = DNN.predict(X)

        # Compute the average prediction
        y_sum = np.empty((len(y_pred_RF)))
        y_mean = np.empty((len(y_pred_RF)))
        for i in range(len(y_pred_RF)):
            y_sum[i] = y_pred_RF[i] + y_pred_GB[i] + y_pred_KNN[i] + y_pred_DNN[i]
            y_mean[i] = y_sum[i]/4

        
        y_pred = y_mean

 
    # get the R2 score
    score = r2_score(y, y_pred)
    # get the MSE
    error = mean_squared_error(y, y_pred)

    print("R2 = " + str(score))
    print("MSE = " + str(error))



def get_coord(address=None):

    print(address)
    # get geolocator
    geolocator = Nominatim(user_agent="boulay")

    # format
    location = ",".join((address).split(",", 2)[:2])

    # get location
    loc = geolocator.geocode(location)

    if loc:
        lat = loc.latitude
        long = loc.longitude
        return lat, long
    else:
        print("The geolocator could not find this address... Please try a different listing")
        exit()

def get_user_input():

    lat = 0
    long = 0
    num_beds = 0
    num_baths = 0
    square = 0
    year = 0
    res = 1.0
    condo = 0.0

    # Ask for address of house
    address = input("What is the listing address? ")

    if(address != ""):
        # Print welcome message if the value is not empty
        lat, long = get_coord(address=address)
    else:
        # Print empty message
        print("The address field can't be empty!")
        exit()

    # ask for number of beds
    beds = input("How many bedrooms does it have? ")
    beds = int(beds)
    if(beds != ""):
        if beds > 0:
            num_beds = beds
        else:
            print("invalid input")
            exit()

    else:
        # Print empty message
        print("The bedrooms field can't be empty!")
        exit()

       # ask for number of bathrooms
    baths = input("How many bathrooms does it have? ")
    baths = int(baths)
    if(baths != ""):
        if baths > 0:
            num_baths = baths
        else:
            print("invalid input")
            exit()
    else:
        # Print empty message
        print("The bathrooms field can't be empty!")
        exit()

    # ask for the squarefootage
    sqrft = input("What's the square footage? ")
    sqrft = int(sqrft)
    if(sqrft != ""):
        if sqrft > 0:
            square = sqrft
        else:
            print("invalid input")
            exit()
    else:
        # Print empty message
        print("The square footage field can't be empty!")
        exit()


    # ask for the year of build
    year_built = input("In what year was the house built? ")
    year_built = int(year_built)
    if(year_built != ""):
        if year_built > 0:
            year = year_built
        else:
            print("invalid input")
            exit()
    else:
        # Print empty message
        print("The year built field can't be empty!")
        exit()

    # ask if its a condo
    cond = input("Is a condo? (answer with either y/n)")
    if(cond != ""):
        # Print welcome message if the value is not empty
        if cond == "y":
            condo = 1.0
            res = 0.0
        elif cond == "n":
            condo = 0.0
            res = 1.0
        else:
            print("invalid input")
            exit()
    else:
        # Print empty message
        print("The condo field can't be empty!")
        exit()
    
    print("latitude: " + str(lat))
    print("longitude: " + str(long))
    print("number of bedrooms: " + str(num_beds))
    print("number of bathrooms: " + str(num_baths))
    print("square footage: " + str(square))
    print("year built: " + str(year))
    print("condo: " + str(condo))

    return [lat,long,0,num_beds,num_baths,square,year,res,condo]


def individual_price_predictor_geo(filename="average"):

    features = get_user_input()

    # Ask for the type of model for estimating
    estimator = "AVG"
    user_estimator = input("Which estimator would you like to use? (RF, GB, KNN, DNN, or AVG) ")

    if(user_estimator != ""):
        # Print welcome message if the value is not empty
        if user_estimator == "RF":
            estimator = user_estimator
        elif user_estimator == "GB":
            estimator = user_estimator
        elif user_estimator == "KNN":
            estimator = user_estimator
        elif user_estimator == "DNN":
            estimator = user_estimator
        elif user_estimator == "AVG":
            estimator = user_estimator
        else:
            print("invalid estimator therefore defaulting to AVG")
        
    else:
        # Print empty message
        print("Please pick.")
        exit()

    # let user know which estimator they chose
    print("The " + str(estimator) + " estimator will be used.")

    # get the test data
    dat = dataProcessing.prepData("../test_set.csv")
    #print(dat.shape)

    data = np.empty((len(dat) + 1, 9))

    # Put in the new data
    data[0] = features
    data[1:] = dat

    # scale the data
    scaled_data, scaler = dataProcessing.scaleData(data)

    #print(scaled_data.shape)

    # Get y and X
    y = scaled_data[:,2]

    # initialize prediction array
    y_pred = np.empty((len(y),))


    X = np.delete(scaled_data, 2, axis=1)
    # Get X data for KNN (lat., long., sqrft, num. bathrooms, and condo)
    X_KNN = np.delete(scaled_data, [2,3,6,7], axis=1)


    # if filename is "average"
    if estimator == "AVG":
        print("running the average of all optimal models")

        # load all models
        RF = pickle.load(open("best_RandomForest.sav", 'rb'))
        GB = pickle.load(open("best_gradientBoost.sav", 'rb'))
        KNN = pickle.load(open("best_knn.sav", 'rb'))
        DNN = tf.keras.models.load_model("best_dnn.h5", compile=False)

        # make all predictions
        y_pred_RF = RF.predict(X)
        y_pred_GB = GB.predict(X)
        y_pred_KNN = KNN.predict(X_KNN)
        y_pred_DNN = DNN.predict(X)

        # Compute the average prediction
        y_sum = np.empty((len(y_pred_RF)))
        y_mean = np.empty((len(y_pred_RF)))
        for i in range(len(y_pred_RF)):
            y_sum[i] = y_pred_RF[i] + y_pred_GB[i] + y_pred_KNN[i] + y_pred_DNN[i]
            y_mean[i] = y_sum[i]/4

        
        y_pred = y_mean

        # make predictions and average
    elif estimator == "RF":
        
        # load the model
        model = pickle.load(open("best_RandomForest.sav", 'rb'))
        # make the prediction
        y_pred = model.predict(X)

    elif estimator == "GB":

        # load the model
        model = pickle.load(open("best_gradientBoost.sav", 'rb'))
        # make the prediction
        y_pred = model.predict(X)

    elif estimator == "KNN":
        # load the model
        model = pickle.load(open("best_knn.sav", 'rb'))
        # make the prediction
        y_pred = model.predict(X_KNN)


    elif estimator == "DNN":
        # load model
        model = tf.keras.models.load_model("best_dnn.h5", compile=False)
        # make the prediction
        y_pred = model.predict(X)
        y_pred = np.reshape(y_pred, (len(y),))


    #print("scaled_data SHAPE: "+str(scaled_data.shape))
    #print("y_pred SHAPE: "+str(y_pred.shape))
    # Rescale the data including the prediction
    pred_dataset = scaled_data
    pred_dataset[:,2] = y_pred

    rescaled_pred_dataset = scaler.inverse_transform(pred_dataset)

    formatted_price = math.floor(rescaled_pred_dataset[0,2])

    #print(rescaled_pred_dataset)
    print("Predicted Price: $" + str(formatted_price))