import pandas as pd

input_file = "./csv/daily_HKO_RF_ALL.csv"  

data = pd.read_csv(input_file)

data['Value'] = data['Value'].replace('Trace', 0.025)
data['Value'] = pd.to_numeric(data['Value'])

complete_data = data[data['Data Completeness'] == 'C']
complete_data['year_month'] = complete_data['Year'].astype(str) + '-' + complete_data['Month'].astype(str).str.zfill(2)

monthly_avg = complete_data.groupby('year_month')['Value'].mean().reset_index()
monthly_avg.columns = ['year_month', 'average_rainfall']

output_file = "./csv/hko_rf_monthly.csv" 
monthly_avg.to_csv(output_file, index=False)

print(f"Monthly average rainfall saved to {output_file}")