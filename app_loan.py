import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/loan_data.csv')
print(data.shape)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

imputer_features = ['person_age','person_emp_exp','loan_int_rate','cb_person_cred_hist_length']
imputer_transformer = SimpleImputer(strategy='mean')

one_hotcoding_features = ['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
one_hotcoding_transformer = OneHotEncoder(drop='first')

scaler_features = ['person_income']
scaler_transformer = StandardScaler()

impute_then_scale_features = ['loan_amnt','credit_score']
impute_then_scale_transformer = Pipeline([
    ('imputer1',SimpleImputer(strategy='mean')),
    ('scaler1',StandardScaler())
])

preprocessors = ColumnTransformer([
    ('imputer',imputer_transformer,imputer_features),
    ('encoder',one_hotcoding_transformer,one_hotcoding_features),
    ('scaler',scaler_transformer,scaler_features),
    ('impute_scaler',impute_then_scale_transformer,impute_then_scale_features)
])

pipeline = Pipeline([
    ('preprocessor',preprocessors),
    ('model',LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
score = accuracy_score(y_test,y_predict)
print('score : ',score)
