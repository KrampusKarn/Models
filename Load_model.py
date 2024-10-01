from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
sena = [2.361737, 0, -0.97559]
name_gcm = list(pd.read_excel('D:\pythonProject\ธีซิส\GCM\\สรุปผล.xlsx')['ชื่อแบบบจำลอง'])
ssp = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
writer = pd.ExcelWriter('จำนวนผู้ติดเชื้อในอนาคต.xlsx')
best_model = joblib.load('D:\pythonProject\ธีซิส\โมเดล\\random\\random_forest_model22.pkl')
for ng in name_gcm:
    for sp in ssp:
        data_load = pd.read_excel('D:\pythonProject\ธีซิส\GCM\\เข้าโมเดล.xlsx', sheet_name=ng + sp)
        save = pd.DataFrame()
        for sn in sena:
            db = [sn for i in range(len(data_load))]
            data_load['dengue_befor'] = pd.Series(db)
            data_load['dengue_befor_1'] = pd.Series(db)
            print(ng)
            X_fut = data_load[['pr', 'pr_1', 'tas', 'tas_1', 'tasmax', 'tasmax_1', 'tasmin', 'tasmin_1', 'deltemp',
                           'deltemp_1', 'dengue_befor', 'dengue_befor_1']]
            save_list = best_model.predict(X_fut)
            print(save_list)
            save[str(sn)] = pd.Series(save_list)
        save.to_excel(writer, sheet_name=ng + sp)
writer._save()