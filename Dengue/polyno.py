import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
def r_squared(y_true, y_pred):

    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
list_parameter = [1.0, 0.27, 6.0, 0.57, 24.0, 34.9, 100.0, 27.9]
# load the data
data = pd.read_csv('dengue_data.csv')
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
X_train = train_data[
    ['temperature', 'precipitation', 'temp_min_binary', 'temp_max_binary', 'deltemp', 'dengue_befor']]
# create the test data
X_test = test_data[
    ['temperature', 'precipitation', 'temp_min_binary', 'temp_max_binary', 'deltemp', 'dengue_befor']]
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

y_test = poly_model.predict(X_test_poly)
y_io_save = []
for i in range(len(y_t_s)):
    y_t_save.append(y_t_s[i][0])
for i in range(len(y_test)):
    if y_pred[i][0] > 0:
        y_pred_save.append(y_pred[i][0])
    else:
        y_pred_save.append(0)
# evaluate the model
y_train_save = y_train['dengue_incidence']
for i in range(len(y_t_s)):
    if y_t_save[i] < 0:
        y_t_save[i] = 0
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_train_save, y_t_save)
r2_test = r_squared(y_test_save,y_pred_save)
bbb = poly_model.coef_
print(bbb)
save = pd.DataFrame()
save['y_train'] = pd.Series(y_train_save)
save['y_t_pre'] = pd.Series(y_t_save)
save['y_test'] = pd.Series(y_test_save)
save['y_pred'] = pd.Series(y_pred_save)
save['coe_f'] = pd.Series(bbb[0])
writer = pd.ExcelWriter('ลิเน1.xlsx', engine='xlsxwriter')
save.to_excel(writer, sheet_name='หน้า1')
writer._save()
print(r2)
print(r2_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared value: {r2_score(y_test, y_pred)}")

