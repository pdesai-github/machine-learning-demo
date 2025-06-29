# from titanic_classification import get_titanic_model_pred
# from titanic_classification_pipeline import get_titanic_model_pred_with_pipeline
import pandas as pd
#
# csv_path = 'data/titanic.csv'
# sample = pd.DataFrame([{
#     'Sex':'female',
#     'Age':45,
#     'Pclass':1
# }])
# pred = get_titanic_model_pred(csv_path,sample)
# print(pred)
#
# sample1 = pd.DataFrame([{
#     'Sex':'male',
#     'Age':45,
#     'Pclass':1
# }])
# pred_with_pipelines = get_titanic_model_pred_with_pipeline(csv_path,sample1)
# print(pred_with_pipelines)
from car_purchace_classification import predict_car_purchase
csv_path = 'data/Social_Network_Ads.csv'

sample_car_purchase = pd.DataFrame([{
    'Gender':'Male',
    'Age':25,
    'EstimatedSalary':10000
}])
sample_pred = predict_car_purchase(csv_path,sample_car_purchase)
print('sample_pred - ',sample_pred)