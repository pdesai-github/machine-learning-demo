import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/Social_Network_Ads.csv')
# print(data.shape)

x = data[['Gender','Age','EstimatedSalary']]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# print(y)

imputer_features = ['Age']
imputer_transformer = SimpleImputer(strategy='mean')

encoding_features = ['Gender']
encoding_transformer = OneHotEncoder(drop='first')

impute_then_scale_features = ['EstimatedSalary']
impute_then_scale_transformer = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

preprocessors = ColumnTransformer([
    ('imputer',imputer_transformer,imputer_features),
    ('encoder',encoding_transformer,encoding_features),
    ('imputer_scaler',impute_then_scale_transformer,impute_then_scale_features)
])

pipeline = Pipeline([
    ('preprocessing',preprocessors),
    ('model',LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train, y_train)
y_predict = pipeline.predict(x_test)
score = accuracy_score(y_test,y_predict)
print('Score : ', score)

sample = pd.DataFrame([{
    'Gender':'Male',
    'Age':40,
    'EstimatedSalary':65000
}])
sample_predict = pipeline.predict(sample)
sample_predict_prob = pipeline.predict_proba(sample)
print('Sampple predict : ',sample_predict)
print('Sampple predict prob : ',sample_predict_prob)


