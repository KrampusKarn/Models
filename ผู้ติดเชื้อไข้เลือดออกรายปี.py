import pandas as pd


sena = [-0.97559]
name_gcm = list(pd.read_excel('D:\pythonProject\ธีซิส\GCM\\สรุปผล.xlsx')['ชื่อแบบบจำลอง'])
ssp = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
save_list = []
sen = ['แย่']
col_save = ['Year','DHF','SSP','Name_GCM','Senario']
for i in range(len(col_save)):
    save_list.append([])
for ng in name_gcm:
    for sp in ssp:
        data_load = pd.read_excel('จำนวนผู้ติดเชื้อในอนาคต.xlsx',sheet_name=ng+sp)
        for i in range(1):
            data = list(data_load[str(sena[i])])
            print(data)
            yy = 2015
            for ii in range(int(len(data)/12)):
                var = 0
                for iii in range(12):
                    var += data[ii*12+iii]
                print(var)
                save_list[0].append(yy+ii)
                save_list[1].append((var-46)/0.46)
                save_list[2].append(sp)
                save_list[3].append(ng)
                save_list[4].append(sen[i])
save = pd.DataFrame()
for i in range(len(col_save)):
    save[col_save[i]] = pd.Series(save_list[i])
save.to_excel('พล็อตกราฟDFH.xlsx')