from titanic_classification import get_titanic_model_pred
import pandas as pd

csv_path = 'data/titanic.csv'
sample = pd.DataFrame([{
    'Sex':'male',
    'Age':45,
    'Pclass':1
}])
pred = get_titanic_model_pred(csv_path,sample)
print(pred)