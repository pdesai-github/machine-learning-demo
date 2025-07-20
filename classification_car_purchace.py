import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def predict_car_purchase(csv_path, sample_car_purchase):
    data = pd.read_csv(csv_path)

    # Features and Labels
    x = data[['Gender','Age','EstimatedSalary']].copy()
    y = data['Purchased'].copy()

    # Train Test split
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)

    # Preprocessing
    categorical_features =['Gender']
    categorical_transformer = OneHotEncoder(drop='first')

    scaler_features = ['Age','EstimatedSalary']
    scaler_transformer = StandardScaler()

    preprocessor = ColumnTransformer([
        ('cat',categorical_transformer,categorical_features),
        ('scaler',scaler_transformer,scaler_features)
    ])

    pipeline = Pipeline([
        ('pre',preprocessor),
        ('model',LogisticRegression())
    ])

    pipeline.fit(x_train,y_train)
    y_predict = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test,y_predict)
    print(accuracy)

    sample_pred = pipeline.predict(sample_car_purchase)
    return  sample_pred