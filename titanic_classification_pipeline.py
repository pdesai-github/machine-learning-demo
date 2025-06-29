import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def get_titanic_model_pred_with_pipeline(csv_path,sample_obj):
    data = pd.read_csv(csv_path)

    # Features and Labels
    x = data[['Sex','Age','Pclass']]
    y = data['Survived']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.8,random_state=42)

    # Preprocessing
    numeric_features = ['Age']
    numeric_transformer = SimpleImputer(strategy='mean')

    categorical_features = ['Sex']
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer([
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('model',LogisticRegression())
    ])

    # Train
    pipeline.fit(x_train,y_train)
    y_predict = pipeline.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test,y_predict)
    report = classification_report(y_test,y_predict)
    print('Accuracy : ', accuracy)
    print(report)

    sample_pred = pipeline.predict(sample_obj)
    return sample_pred


