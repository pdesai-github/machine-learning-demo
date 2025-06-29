from titanic_classification import get_titanic_model_pred
from titanic_classification_pipeline import get_titanic_model_pred_with_pipeline
import pandas as pd

csv_path = 'data/titanic.csv'
sample = pd.DataFrame([{
    'Sex':'female',
    'Age':45,
    'Pclass':1
}])
pred = get_titanic_model_pred(csv_path,sample)
print(pred)

sample1 = pd.DataFrame([{
    'Sex':'male',
    'Age':45,
    'Pclass':1
}])
pred_with_pipelines = get_titanic_model_pred_with_pipeline(csv_path,sample1)
print(pred_with_pipelines)