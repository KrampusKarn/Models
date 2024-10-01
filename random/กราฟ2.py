import pandas as pd
import matplotlib.pyplot as plt

# Your data
df = pd.read_excel('D:\ธีซิส\รูป\\CDF.xlsx',sheet_name='Sheet1')
# Convert data to DataFrame
print(df)
# Plot each column
for column in df.columns:
    if column == 'X':
        continue
    elif column == 'OBS':
        df.plot(x='X', y=column, label=column, linewidth=2.5,ax=plt.gca())
    else:
        df.plot(x='X', y=column, label=column, linewidth=0.5, color='grey', alpha=0.5, ax=plt.gca())

plt.title('Frequency distribution graph of data')
plt.ylabel('Frequency distribution')
plt.xlabel('Precipitation (mm/month)')
plt.legend(loc='best')
plt.grid(True)
plt.show()