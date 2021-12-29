import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import urllib.request
import json
from time import strftime
import dill

dill._dill._reverse_typemap['ClassType'] = type


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return str(obj)


def get_prediction(request_dict):
    body = request_dict
    # для работы на локальном компьютере
    # myurl = "http://10.6.123.58:8180/predict"
    # myurl = "http://192.168.1.68:8180/predict"

    # для работы с докер контейнером
    myurl = "http://localhost:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body, cls=NpEncoder)
    jsondataasbytes = jsondata.encode('utf-8')  # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    # print (jsondataasbytes)
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']


# пути для локальной загрузки примера с тестовыми данными
test_df_path = 'test.csv'

test_df = pd.read_csv(test_df_path, index_col=0)

X = test_df.drop(columns='not.fully.paid')
y = test_df['not.fully.paid']
# разобъем датасет
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)

# для проверки работы API с единичным обращением возьмем первую строку тестового датасета
data2req = dict(X_test.iloc[0, :])
print('Для прогноза невозврата кредита используем эти данные:')
print(data2req)

print(f"Вероятность невозврата: \n{get_prediction(data2req)}")

print()

print("Произведем массовую оценку невозврата для некоторого количества клиентов")
print(f'Начали в: \t\t{strftime("[%Y-%b-%d %H:%M:%S]")}')
predictions = X_test.apply(lambda x: get_prediction(dict(x)), 1)
print(f'Закончили в: \t{strftime("[%Y-%b-%d %H:%M:%S]")}')
print()
print("Оценка качества работы модели:")
print(f"ROC_AUC score: {roc_auc_score(y_score=predictions.values, y_true=y_test)}")
