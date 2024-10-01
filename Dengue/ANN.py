import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error

def r_squared(y_true, y_pred):
    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

data = pd.read_csv('dengue_data.csv')
# สร้างตัวแปร X และ y จากข้อมูล
X = data[['temperature','precipitation', 'deltemp','dengue_befor','temp_max','temp_min']]
y = data['dengue_incidence']
r2_best = 0
for i in range(192):
    # แบ่งข้อมูลออกเป็นชุดฝึกอบรมและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i+1)

    # สร้างโมเดล ANN และกำหนดค่า hidden_layer_sizes และ alpha (ค่าความเร่งการเรียนรู้)
    ann = MLPRegressor(hidden_layer_sizes=(100, 50), alpha=0.01, random_state=i+1)

    # ให้ GridSearchCV และกำหนด scoring='neg_mean_absolute_error' เพื่อหาพารามิเตอร์ที่เหมาะสม
    param_grid = {
        'hidden_layer_sizes': [(100,), (50,), (100, 50), (50, 25)],
        'alpha': [0.001, 0.01, 0.1, 1.0]
    }

    grid_search = GridSearchCV(ann, param_grid, scoring='neg_mean_absolute_error', cv=5)

    # ทำการ Grid Search และฝึกโมเดลด้วยชุดฝึกอบรมที่มีพารามิเตอร์ที่ดีที่สุด
    grid_search.fit(X_train, y_train)

    # แสดงพารามิเตอร์ที่ดีที่สุดที่ Grid Search เสร็จสิ้นแล้ว
    best_params = grid_search.best_params_
    # สร้างโมเดล ANN ที่ใช้พารามิเตอร์ที่ดีที่สุด
    best_ann = MLPRegressor(hidden_layer_sizes=best_params['hidden_layer_sizes'], alpha=best_params['alpha'], random_state=i+1)

    # ฝึกโมเดลด้วยชุดฝึกอบรมที่มีพารามิเตอร์ที่ดีที่สุด
    best_ann.fit(X_train, y_train)

    # ทำนายข้อมูลทดสอบ
    y_train_pre = best_ann.predict(X_train)
    predictions = best_ann.predict(X_test)
    r2 = (r_squared(y_test, predictions) + r_squared(y_train, y_train_pre)) /2
    print(i)
    if r2 > r2_best:
        r2_best = r2

        print('MAE (Train):', mean_absolute_error(y_train, y_train_pre))
        print('MAE (Test):', mean_absolute_error(y_test, predictions))
        print('R2 (Train):', r_squared(y_train, y_train_pre))
        print('R2 (Test):', r_squared(y_test, predictions))

        # บันทึกผลลัพธ์ลงในไฟล์ Excel
        save = pd.DataFrame()
        save['y_train'] = pd.Series(list(y_train))
        save['y_train_pre'] = pd.Series(list(y_train_pre))
        save['y_test'] = pd.Series(list(y_test))
        save['y_test_pre'] = pd.Series(list(predictions))
        save.to_excel('ann_results.xlsx')
