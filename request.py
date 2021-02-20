import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'LIMIT_BAL':20000,'EDUCATION':2,'MARRIAGE':2,'AGE':25,'PAY_1':2,'BILL_AMT1':3913,'BILL_AMT2':3176,'BILL_AMT3':0,'BILL_AMT4':3462,'BILL_AMT5':0,'BILL_AMT6':0,'PAY_AMT1':324,'PAY_AMT2':355,'PAY_AMT3':0,'PAY_AMT4':0,'PAY_AMT5':0,'PAY_AMT6':0})

print(r.json())
