import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


import numpy as np


def r_squared(y_true, y_pred):
    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2



best_r2 = 0
best_params = {}

# Define the range of values for each parameter
befor_yok_range = np.linspace(0, 10, 11)
temp_yok_range = np.linspace(0, 10, 11)
cc_range = np.linspace(0, 10, 11)
temp_up_range = np.linspace(20, 30, 11)
jj_range = np.linspace(0, 10, 11)
# Iterate over all combinations of parameter values
opt = 1

r2 = 0

def dange(list_parameter,i):
    list_parameter[i] = round(list_parameter[i],3)
    x = 0
    if list_parameter[i] < 0:
        r2 = 0
    else:
        data = pd.read_csv('dengue_data1.csv')
        data['dengue_befor'] = ((data['dengue_befor'] - sum(data['dengue_befor']) / len(data['dengue_befor'])) / sum(
            data['dengue_befor']) * len(data['dengue_befor']))
        data['temperature'] = (((data['temperature'] - list_parameter[7]) ** 2) ** 0.5) ** list_parameter[1]
        data['precipitation'] = (((data['precipitation']) ** 2) ** 0.5) ** list_parameter[2]
        data['deltemp'] = (data['temp_max'] - data['temp_min']) ** list_parameter[3]
        data['temp_min_binary'] = (data['temp_min'] > list_parameter[4]).astype(int)
        data['temp_max_binary'] = (data['temp_max'] < list_parameter[5]).astype(int)
        # Handle missing values in the dataset
        imputer = SimpleImputer()
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        # split the data into training and test sets
        train_data = data[:]
        test_data = data[int(0.7 * len(data)):]
        # create the training data
        X_train = train_data[['temperature', 'precipitation', 'temp_min_binary', 'temp_max_binary', 'deltemp','dengue_befor']]
        # create the test data
        X_test = test_data[['temperature', 'precipitation', 'temp_min_binary', 'temp_max_binary', 'deltemp','dengue_befor']]
        y_train = train_data[['dengue_incidence']]
        y_test = test_data[['dengue_incidence']]

        # create the polynomial regression model
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        #X_train_poly = X_train
        #X_test_poly = X_test
        poly_model = LinearRegression()

        # fit the model to the training data
        poly_model.fit(X_train_poly, y_train)

        # make predictions on the test data
        y_pred = poly_model.predict(X_test_poly)

        # evaluate the model
        y_test_save = list(y_test['dengue_incidence'])
        y_pred_save = []
        y_t_save = []
        y_t_s = poly_model.predict(X_train_poly)
        for ip in range(len(y_t_s)):
            y_t_save.append(y_t_s[ip][0])
        for ip in range(len(y_test)):
            if y_pred[ip][0] > 0:
                y_pred_save.append(y_pred[ip][0])
            else:
                y_pred_save.append(0)
        # evaluate the model
        y_train_save = y_train['dengue_incidence']
        for ip in range(len(y_t_s)):
            if y_t_save[ip] < 0:
                y_t_save[ip] = 0
        mse = mean_squared_error(y_test, y_pred)
        r2 = r_squared(y_train_save, y_t_save)
    return r2
rmax = 0
r2_now = 0
list_save = []
r2_list = []
import random
list_parameter = [1.0, 0.037000000000000005, 5.78, 0.29000000000000004, 23.76, 34.24, 100.0, 27.1]
for asdfgh in range(1):
    opt = 1
    for iadsd in range(3):
        for ii in range(1):
            for iii in range(len(list_parameter)):
                i = iii
                list_parameter[i] -= opt
                r2_befor = dange(list_parameter,i)
                list_parameter[i] += opt
                r2_now = dange(list_parameter,i)
                list_parameter[i] += opt
                r_next = dange(list_parameter,i)
                list_parameter[i] -= opt
                while r2_befor > r2_now or r_next > r2_now:
                    if r2_befor > r2_now:
                        list_parameter[i] = list_parameter[i] - opt
                    else:
                        list_parameter[i] += opt
                    list_parameter[i] -= opt
                    r2_befor = dange(list_parameter,i)
                    list_parameter[i] += opt
                    r2_now = dange(list_parameter,i)
                    list_parameter[i] += opt
                    r_next = dange(list_parameter,i)
                    list_parameter[i] -= opt
        opt /= 10
list_parameter = [1.0, 0.067, 2.12, 0.18, 24.66, 33.339999999999996, 100.0, 27.099999999999998]
data = pd.read_csv('dengue_data1.csv')
data['dengue_befor'] = ((data['dengue_befor'] - sum(data['dengue_befor']) / len(data['dengue_befor'])) / sum(
    data['dengue_befor']) * len(data['dengue_befor']))
data['temperature'] = (((data['temperature'] - list_parameter[7]) ** 2) ** 0.5) ** list_parameter[1]
data['precipitation'] = (((data['precipitation']) ** 2) ** 0.5) ** list_parameter[2]
data['deltemp'] = (data['temp_max'] - data['temp_min']) ** list_parameter[3]
data['temp_min_binary'] = (data['temp_min'] > list_parameter[4]).astype(int)
data['temp_max_binary'] = (data['temp_max'] < list_parameter[5]).astype(int)
# Handle missing values in the dataset
imputer = SimpleImputer()
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# split the data into training and test sets
train_data = data[:int(0.7 * len(data))]
test_data = data[int(0.7 * len(data)):]
# create the training data
X_train = train_data[['temperature', 'precipitation', 'temp_min_binary', 'temp_max_binary', 'deltemp','dengue_befor']]
# create the test data
X_test = test_data[['temperature', 'precipitation', 'temp_min_binary', 'temp_max_binary', 'deltemp','dengue_befor']]
y_train = train_data[['dengue_incidence']]
y_test = test_data[['dengue_incidence']]

# create the polynomial regression model
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
#X_train_poly = X_train
#X_test_poly = X_test
poly_model = LinearRegression()
for i in range(len(X_train_poly)):
    print(X_train_poly[i])
# fit the model to the training data
poly_model.fit(X_train_poly, y_train)

# make predictions on the test data
y_pred = poly_model.predict(X_test_poly)

# evaluate the model
y_test_save = list(y_test['dengue_incidence'])
y_pred_save = []
y_t_save = []
y_t_s = poly_model.predict(X_train_poly)
for ip in range(len(y_t_s)):
    y_t_save.append(y_t_s[ip][0])
for ip in range(len(y_test)):
    if y_pred[ip][0] > 0:
        y_pred_save.append(y_pred[ip][0])
    else:
        y_pred_save.append(0)
# evaluate the model
y_train_save = y_train['dengue_incidence']
for ip in range(len(y_t_s)):
    if y_t_save[ip] < 0:
        y_t_save[ip] = 0
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_train_save, y_t_save)
print(poly_model.coef_)
poly_coef = poly_model.coef_
poly_coef_list = []
for i in range(len(poly_coef[0])):
    print(poly_coef[0][i])
    poly_coef_list.append(poly_coef[0][i])
save = pd.DataFrame()
save['train'] = pd.Series(list(y_train_save))
save['Y_train'] = pd.Series(list(y_t_save))
save['poly_coef'] = pd.Series(list(poly_coef_list))
save.to_excel('ln5.xlsx')
print(list_parameter)