import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/titanic.csv')

x = data[['Pclass','Sex','Age','Cabin']].copy()
y = data.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

encoding_features = ['Sex']
encoding_transformer = OneHotEncoder()

imputing_features = ['Age', 'Pclass']
imputing_transformer = SimpleImputer(strategy='mean')

imputing_then_encoding_features = ['Cabin']
imputing_then_encoding_transformer = Pipeline([
    ('imputing',SimpleImputer(strategy='most_frequent')),
    ('encoding',OneHotEncoder(handle_unknown='ignore'))
])

preprocessing = ColumnTransformer([
    ('encoding', encoding_transformer, encoding_features),
    ('imputing', imputing_transformer, imputing_features),
    ('imputing_then_encoding', imputing_then_encoding_transformer, imputing_then_encoding_features)
])

pipeline = Pipeline([
    ('preprocessing',preprocessing),
    ('model',LogisticRegression())
])

pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
score = accuracy_score(y_test,y_predict)

sample =  pd.DataFrame([{
    'Pclass': 3,
    'Sex': 'female',
    'Age': 25,
    'Cabin': 'C123'
}])
sample_prediction = pipeline.predict(sample)
print(sample_prediction)