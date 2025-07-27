import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

data = pd.read_csv('data/regression/home-prices.csv')
print(data.shape)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)

text_features = ['Area']
text_features_transformer = OneHotEncoder()

impute_then_scale_features = ['Size','DistancefromCityCenter']
impute_then_scale_transformer = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

preprocessing = ColumnTransformer([
    ('text',text_features_transformer,text_features),
    ('impute_scale',impute_then_scale_transformer,impute_then_scale_features)
])

pipeline = Pipeline([
    ('preprocessing',preprocessing),
    ('model',LinearRegression())
])

pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
r2 = r2_score(y_test,y_predict)
print(r2)

train_score = pipeline.score(x_train, y_train)
test_score = pipeline.score(x_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R² : {test_score:.4f}")




