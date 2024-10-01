import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

def r_squared(y_true, y_pred):
    """Compute the R-squared value for two arrays of data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

data = pd.read_csv('dengue_data.csv')
# สร้างตัวแปร X และ y จากข้อมูล
""
X = data[['temperature', 'precipitation', 'deltemp', 'dengue_befor', 'temp_max', 'temp_min'
          ]]
y = data['dengue_incidence']

# แบ่งข้อมูลออกเป็นชุดฝึกอบรมและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# ปรับค่าของ X และ y ให้อยู่ในรูปของ sequence สำหรับ RNN
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# กำหนดความยาวของ Time Series ที่คุณต้องการ (ในกรณีนี้คือ 12 เดือน)
time_steps = 12

# สร้าง Time Series DataFrame โดยใช้ข้อมูล X ที่ปรับค่าแล้ว
def create_time_series_data(X, y, time_steps):
    X_time_series, y_time_series = [], []
    for i in range(len(X) - time_steps):
        X_time_series.append(X[i: i + time_steps])
        y_time_series.append(y[i + time_steps])
    return np.array(X_time_series), np.array(y_time_series)

X_train_time_series, y_train_time_series = create_time_series_data(X_train_scaled, y_train.values, time_steps)
X_test_time_series, y_test_time_series = create_time_series_data(X_test_scaled, y_test.values, time_steps)

# สร้างโมเดล RNN
model = Sequential()
model.add(LSTM(24, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])))
model.add(Dense(1))

model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# ฝึกโมเดลด้วยชุดฝึกอบรม
model.fit(X_train_time_series, y_train_time_series, epochs=300, batch_size=1)

# ทำนายข้อมูลทดสอบ
y_train_pre = model.predict(X_train_time_series)
predictions = model.predict(X_test_time_series)

print('MAE (Train):', mean_absolute_error(y_train_time_series, y_train_pre))
print('MAE (Test):', mean_absolute_error(y_test_time_series, predictions))
print('R2 (Train):', r_squared(y_train_time_series, y_train_pre))
print('R2 (Test):', r_squared(y_test_time_series, predictions))

# บันทึกผลลัพธ์ลงในไฟล์ Excel
save = pd.DataFrame()
save['y_train'] = pd.Series(list(y_train_time_series))
save['y_train_pre'] = pd.Series(list(y_train_pre.flatten()))
save['y_test'] = pd.Series(list(y_test_time_series))
save['y_test_pre'] = pd.Series(list(predictions.flatten()))
save.to_excel('rnn_results.xlsx')
