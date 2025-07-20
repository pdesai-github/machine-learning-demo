import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/Social_Network_Ads.csv')

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42)

impute_encode_features = ['Gender']
impute_encode_transformer = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop='first'))
])

impute_features = ['Age']
impute_transformer = SimpleImputer(strategy='most_frequent')

impute_scale_features = ['EstimatedSalary']
impute_scale_transformer = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
])

preprocessor = ColumnTransformer([
    ('imp_enc',impute_encode_transformer,impute_encode_features),
    ('imp',impute_transformer,impute_features),
    ('imp_scal',impute_scale_transformer,impute_scale_features)
])

pipeline = Pipeline([
    ('pre',preprocessor),
    ('model',LogisticRegression())
])
pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
score = accuracy_score(y_test,y_predict)
print(score)

sample = pd.DataFrame([{
    'Gender': 'Male',
    'Age': 30,
    'EstimatedSalary': 150000
}])
sample_predict = pipeline.predict(sample)
if sample_predict == 1:
    print('User may purchase')
else:
    print('User may not purchase')

