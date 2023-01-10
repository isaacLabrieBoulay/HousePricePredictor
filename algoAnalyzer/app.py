# we need to make an app that can take in user input

# print("Please enter for the following:")
# print("")
# lat = input("Latitide: ")
# lng = input("Longitude: ")
# beds = input("Number of Bedrooms: ")
# bath = input("Number of Bathrooms: ")
# sqFt = input("Square footage: ")
# year = input("Year of build: ")
# typeOfComplex = input("1 = Residential, 0 = Condo: ")

# residential = 0
# condo = 0

# if typeOfComplex == "1":
#     residential = 1
# else:
#     condo = 1

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# test_data = [[float(lat), float(lng), float(beds), float(bath), int(sqFt), float(year), float(residential), float(condo)]]

# print(test_data)

# load the model
import keras

my_model = keras.models.load_model("best_model.h5", custom_objects={"r_square": r_square}) 

sup = [[50.420990, -104.511124, 4.0, 3.0, 5565, 2014, 1.0, 0.0]]

expected_price = my_model.predict(sup)

print("Expected price: $" + str(expected_price))
from sklearn.metrics import r2_score
final = expected_price[0][0]
print(final)

true_value = 609900.0
one = (true_value - final)
two = (one / true_value) * 100
percent_accuracy = 100 - two
print("Accuracy: %" + str(percent_accuracy))



# 50.420990
# -104.511124
# 4 bed
# 3 bath
# 5565

# 50.411480
# -104.521670
# 3
# 3
# 5644
# 2014