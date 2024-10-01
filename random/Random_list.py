from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import gamma
def quantile_mapping(target_data, reference_data):
    target_quantiles = np.percentile(target_data, np.arange(0, 100))
    reference_quantiles = np.percentile(reference_data, np.arange(0, 100))

    target_gamma_params = gamma.fit(target_data)
    reference_gamma_params = gamma.fit(reference_data)

    quantile_mapped = gamma.ppf(np.interp(gamma.cdf(target_data, *target_gamma_params), gamma.cdf(reference_data, *reference_gamma_params), reference_quantiles))
    return quantile_mapped

def r_squared(y_true, y_pred):

    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
r2_best = 0
r_2_list = []
writer = pd.ExcelWriter('โมเดล2.xlsx',engine='xlsx')
save = pd.DataFrame()
save_1 = pd.DataFrame()
save_2 = pd.DataFrame()
save_3 = pd.DataFrame()
save_4 = pd.DataFrame()
save_5 = pd.DataFrame()
for i in range(0,192):
    data = pd.read_csv('dengue_data.csv')
    # สร้างตัวแปร X และ y จากข้อมูล
    #
    X = data[['temperature','precipitation', 'deltemp','dengue_befor','temp_max','temp_min']]
    y = data['dengue_incidence']

    # แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i+1)

    # กำหนดช่วงค่าที่ต้องการทดสอบในพารามิเตอร์ของ Random Forest
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': [1]
    }

    # สร้างโมเดล Random Forest
    rf = RandomForestRegressor(random_state=i+1)

    # ใช้ GridSearchCV ในการหาพารามิเตอร์ที่ดีที่สุด
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
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
    print(r2)
    r_2_list.append(r2)
    print(i)
    r2_best = r2
    y_train_pre = best_model.predict(X_train)
    print('MAE:', mean_absolute_error(y_train, y_train_pre))
    print('MAE:', mean_absolute_error(y_test, predictions))
    print('R2:', r_squared(y_train, y_train_pre))

    print('R2:',r_squared(y_test,predictions))

    lum = []
    for iiii in range(192):
        lum.append(iiii)
    save['y_train'+str(i)] = pd.Series(list(y_train))
    save_1['y_train_pre'+str(i)] = pd.Series(list(y_train_pre))
    save_2['y_test'+str(i)] = pd.Series(list(y_test))
    save_3['y_test_pre'+str(i)] = pd.Series(list(predictions))
    feature_importances = best_model.feature_importances_
    save_4['y_test_pre'+str(i)] = pd.Series(list(predictions))
    save_5['feature_importances'+str(i)] = pd.Series(feature_importances)
    print(feature_importances)
save.to_excel(writer,sheet_name='1')
save_1.to_excel(writer,sheet_name='2')
save_2.to_excel(writer,sheet_name='3')
save_3.to_excel(writer,sheet_name='4')
save_4.to_excel(writer,sheet_name='5')
save_5.to_excel(writer,sheet_name='6')
save_6 = pd.DataFrame()
save_6['r_2_list'] = pd.Series(r_2_list)
save_6.to_excel(writer,sheet_name='7')
writer._save()
# Apply quantile mapping to the predictions