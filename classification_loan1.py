import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/loan_data.csv')

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42)

imputing_numeric_features = ['person_age','person_income','person_emp_exp','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score']
imputing_numeric_transformer = SimpleImputer(strategy='mean')

imputing_encoding_text_features =['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
imputing_encoding_text_transformer = Pipeline([
    ('imputing',SimpleImputer(strategy='most_frequent')),
    ('encoding',OneHotEncoder())
])

scaling_features = ['person_income','loan_amnt','loan_int_rate']
scaling_transformer = StandardScaler()

preprocessing = ColumnTransformer([
    ('num_imp',imputing_numeric_transformer,imputing_numeric_features),
    ('imp_enc',imputing_encoding_text_transformer,imputing_encoding_text_features),
    ('scaling',scaling_transformer, scaling_features)
])

pipeline = Pipeline([
    ('pre',preprocessing),
    ('model',LogisticRegression(max_iter=1000))
])
pipeline.fit(x_train,y_train)
y_predict = pipeline.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(score)

sample = pd.DataFrame([
    {
        'person_age': 21,
        'person_gender': 'female',
        'person_education': 'High School',
        'person_income': 12282,
        'person_emp_exp': 0,
        'person_home_ownership': 'OWN',
        'loan_amnt': 1000,
        'loan_intent': 'EDUCATION',
        'loan_int_rate': 11.14,
        'loan_percent_income': 0.08,
        'cb_person_cred_hist_length': 2,
        'credit_score': 504,
        'previous_loan_defaults_on_file': 'Yes'
    }
])
print('Sample prediction:', pipeline.predict(sample))

