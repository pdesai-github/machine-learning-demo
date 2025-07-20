import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('data/Salary_Data.csv')

x = data[['YearsExperience']]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42)

impute_then_scale_features = ['YearsExperience']
impute_then_scale_transformer = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
])

preprocesser = ColumnTransformer([
    ('impute_scale',impute_then_scale_transformer,impute_then_scale_features)
])

pipeline = Pipeline([
    ('preprocessor',preprocesser),
    ('model',LinearRegression())
])
pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
score = r2_score(y_test,y_predict)
print(score)

sample =  pd.DataFrame([{
    "YearsExperience" : 13
}])
sample_pred = pipeline.predict(sample)
print('Salary predicted - ',sample_pred)