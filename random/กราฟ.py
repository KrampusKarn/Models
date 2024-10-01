import matplotlib.pyplot as plt
import pandas as pd
# ข้อมูล
data_load = pd.read_excel('random20.xlsx')
dates = list(data_load['date'])  # ใส่วันที่ที่คุณมีในรายการ
y_train = list(data_load['q'][:143])  # ใส่ค่า y_train ที่คุณมีในรายการ
y_train_model = list(data_load['w'][:143])  # ใส่ค่า y_test ที่คุณมีในรายการ
y_test = list(data_load['e'][:48])  # ใส่ค่า y_train_model ที่คุณมีในรายการ
y_test_model = list(data_load['r'][:48])  # ใส่ค่า y_test_model ที่คุณมีในรายการ

# สร้างกราฟ
plt.figure(figsize=(15, 7))
plt.plot(dates, y_train + y_test, label='Actual Values', color='blue')
plt.plot(dates, y_train_model + y_test_model, label='Model Predictions', color='red', linestyle='--')

# ตั้งชื่อแกน x และ y
plt.xlabel('Date')
plt.ylabel('Value')

# แสดง legend
plt.legend()

# ตั้งชื่อกราฟ
plt.title('Comparison between Actual and Model Predicted Values')

# แสดงกราฟ
plt.xticks(rotation=45)  # หมุนแกน x ให้เป็น 45 องศาเพื่อความชัดเจน
plt.tight_layout()
plt.show()
