import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def prepData(fileName):
    df = pd.read_csv(fileName)
    df.columns = ['index', 'latitude', 'longitude', 'price', 'beds', 'bath', 'square', 'year', 'residential', 'condo']
    df.drop('index', inplace=True, axis=1)
    filtered_df = df.dropna()
    returnedArray = filtered_df.to_numpy()
    return returnedArray

def scaleData(numpyArr):
    scaler = preprocessing.MinMaxScaler()
    scaledData = scaler.fit_transform(numpyArr)
    return (scaledData, scaler)
    
def split(scaledData):
    #price
    y = scaledData[:, 2]

    #Parameters (x)
    scaled_woPrice = np.delete(scaledData, 2, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(scaled_woPrice, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

def split_KNN(scaledData):
    #price
    y = scaledData[:, 2]

    #Parameters (x)
    scaled_woFeatures = np.delete(scaledData, [2,3,6,7], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(scaled_woFeatures, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
