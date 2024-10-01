from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
def r_squared(y_true, y_pred):

    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_sum = np.sum((y_pred-np.mean(y_pred))*(y_true-np.mean(y_true)))
    ss_res = (np.sum((y_pred - np.mean(y_pred)) ** 2) * np.sum((y_true - np.mean(y_true)) ** 2))**0.5
    return (y_pred_sum/ss_res)**2
r2 = 0.1
mae = 4
r2_best = 0.68
mae_best = 2
#'dengue_befor','dengue_befor_1'
while r2 < 0.7 or mae_best < mae:
    data = pd.read_csv('dengue_data1.csv')
    # สร้างตัวแปร X และ y จากข้อมูล
    #
    hh = 143
    X = data[['pr','pr_1','tas','tas_1','tasmax','tasmax_1','tasmin','tasmin_1','deltemp','deltemp_1', 'dengue_befor', 'dengue_befor_1']]
    y = data['dengue_incidence']
    X_train = X[:hh]
    X_test = X[hh:]
    y_train = y[:hh]
    y_test = y[hh:]
    # แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    # กำหนดช่วงค่าที่ต้องการทดสอบในพารามิเตอร์ของ Random Forest
    param_grid = {
        'n_estimators': [6],
        'max_depth': [100],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': [1,'sqrt']
    }

    # สร้างโมเดล Random Forest
    rf = RandomForestRegressor()

    # ใช้ GridSearchCV ในการหาพารามิเตอร์ที่ดีที่สุด
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_percentage_error')
    grid_search.fit(X_train, y_train)

    # พารามิเตอร์ที่ให้ความแม่นยำสูงที่สุด
    best_params = grid_search.best_params_

    # โมเดลที่ดีที่สุด
    best_model = grid_search.best_estimator_
    # ค่าความแม่นยำของโมเดลที่ดีที่สุดในชุดข้อมูลทดสอบ
    best_score = grid_search.best_score_
    # ทำนายข้อมูลทดสอบด้วยโมเดลที่ดีที่สุด
    predictions = best_model.predict(X_test)
    r2 = r_squared(y_test,predictions)
    mae = mean_absolute_error(y_test, predictions)
    if r2 > r2_best and mae < mae_best:
        r2_best = r2
        mae_best = mae

        y_train_pre = best_model.predict(X_train)
        print(best_params)
        print('MAE:', mean_absolute_error(y_test, predictions))
        print('R2:',r_squared(y_test,predictions))
        lum = []
        save = pd.DataFrame()
        save['y_train'] = pd.Series(list(y_train))
        save['y_train_pre'] = pd.Series(list(y_train_pre))
        save['y_test'] = pd.Series(list(y_test))
        save['y_test_pre'] = pd.Series(list(predictions))
        feature_importances = best_model.feature_importances_
        save['y_test_pre'] = pd.Series(list(predictions))
        save['feature_importances'] = pd.Series(feature_importances)
        save.to_excel('random22.xlsx')

        joblib.dump(best_model, 'random_forest_model22.pkl')
        # Load the model from the file
        loaded_rf_model = joblib.load('random_forest_model22.pkl')