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
tmp = 0
r2max = 0
for tmpower in range(10):
    for prpower in range(10):
        for delpower in range(10):
            for tmin in range(10):
                print('tmpower' + str(tmpower) + 'prpower' + str(prpower) + 'delpower' + str(delpower) + 'tmin' + str(
                    tmin))
                for tmax in range(10):
                    for tmp in range(10):
                        data = pd.read_csv('dengue_data1.csv')
                                # Handle missing values in the dataset
                        data['dengue_befor'] = ((data['dengue_befor'] - sum(data['dengue_befor']) / len(data['dengue_befor'])) / sum(
                            data['dengue_befor']) * len(data['dengue_befor']))
                        data['temperature'] = (((data['temperature'] - (tmp*1.25+19.9)) ** 2) ** 0.5) ** (tmpower*0.1+0.1)
                        data['precipitation'] = (((data['precipitation']) ** 2) ** 0.5) ** (prpower*1+1)
                        data['deltemp'] = (data['temp_max'] - data['temp_min']) ** (delpower*1+1)
                        data['temp_min'] = (data['temp_min'] > (tmin*1.26*+13.255)).astype(int)
                        data['temp_max'] = (data['temp_max'] < (tmax*1.2+27.16)).astype(int)
                        imputer = SimpleImputer()
                        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
                        # split the data into training and test sets
                        train_data = data[:]
                        test_data = data[int(0.7 * len(data)):]
                        # create the training data
                        X_train = train_data[['temperature', 'precipitation', 'temp_min', 'temp_max', 'deltemp','dengue_befor']]
                        # create the test data
                        X_test = test_data[['temperature', 'precipitation', 'temp_min', 'temp_max', 'deltemp','dengue_befor']]
                        y_train = train_data[['dengue_incidence']]
                        y_test = test_data[['dengue_incidence']]

                        # create the polynomial regression model
                        poly_features = PolynomialFeatures(degree=2)
                        X_train_poly = poly_features.fit_transform(X_train)
                        X_test_poly = poly_features.transform(X_test)
                        X_train_poly = X_train
                        X_test_poly = X_test
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
                        if r2 > r2max:
                            r2max = r2
                            parameter = []
                            parameter.append(tmp*0.125+19.9)
                            parameter.append(tmpower*0.1+0.1)
                            parameter.append(prpower*0.1+0.1)
                            parameter.append(delpower*0.1+0.1)
                            parameter.append(tmin*0.126+13.255)
                            parameter.append(tmax*0.12+27.16)
                            print(r2max)
                            save = pd.DataFrame()
                            save['y_train'] = pd.Series(list(y_train_save))
                            save['y_train_pre'] = pd.Series(list(y_t_save))
                            save['parameter'] = pd.Series(list(parameter))
                            save.to_excel("ln.xlsx")