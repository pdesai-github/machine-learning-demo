import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data  = pd.read_csv('data/student_classification_data.csv')

x = data.iloc[:,1:4]
y = data.iloc[:,-1]

# Clean the Result column
y = data['Result'].str.strip()
print('Unique Result values:', y.unique())
le = LabelEncoder()
y = le.fit_transform(y)
print('Encoded y:', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

imputing_features = ['Age','StudyHour']
imputing_transformer = SimpleImputer(strategy='mean')


imputing_then_scalling_features = ['Attendance']
imputing_then_scalling_transformer = Pipeline([
    ('imputing',SimpleImputer(strategy='mean')),
    ('scalling',StandardScaler())
])

preprocessing = ColumnTransformer([
    ('imputing',imputing_transformer,imputing_features),
    ('imputing_then_scalling',imputing_then_scalling_transformer,imputing_then_scalling_features)
])

pipeline = Pipeline([
    ('preprocessor',preprocessing),
    ('model',LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('Score : ',score)
class_rep = classification_report(y_test, y_predict)

sample = pd.DataFrame([{
    'Age':18,
    'StudyHour':5,
    'Attendance':20
}])
sample_pred = pipeline.predict(sample)
print(sample_pred)
