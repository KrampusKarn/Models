import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

data = pd.read_csv('dengue_data.csv')
def r_squared(y_true, y_pred):
    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

X = data[['temperature', 'precipitation', 'deltemp', 'dengue_befor', 'temp_max', 'temp_min']]
y = data['dengue_incidence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k_values = [1, 3, 5, 7, 9]

best_k = None
best_mae = float('inf')

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"K = {k}, MAE = {mae}")

    if mae < best_mae:
        best_k = k
        best_mae = mae
knn = KNeighborsRegressor(n_neighbors=best_k)

knn.fit(X_train, y_train)

y_train_pre = knn.predict(X_train)
predictions = knn.predict(X_test)

save = pd.DataFrame()
save['y_train'] = pd.Series(list(y_train))
save['y_train_pre'] = pd.Series(list(y_train_pre))
save['y_test'] = pd.Series(list(y_test))
save['y_test_pre'] = pd.Series(list(predictions))
print(list(y_test))
print(list(predictions))

print('MAE:', mean_absolute_error(y_train, y_train_pre))
print('MAE:', mean_absolute_error(y_test, predictions))
print('R2:', r_squared(y_train, y_train_pre))
print('R2:', r_squared(y_test, predictions))

save.to_excel('knn_results.xlsx')
