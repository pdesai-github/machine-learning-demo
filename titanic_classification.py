import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def get_titanic_model_pred(csv_path, input_obj):
    data = pd.read_csv(csv_path)

    # Split Features and Labels
    x = data[['Sex', 'Age', 'Pclass']].copy()
    y = data['Survived'].copy()

    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    # Fill missing data
    x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
    x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())

    # Encoding
    le = LabelEncoder()
    x_train['Sex'] = le.fit_transform(x_train['Sex'])
    x_test['Sex'] = le.transform(x_test['Sex'])

    # Train
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_predict)
    report = classification_report(y_test, y_predict)

    input_obj['Sex'] = le.transform(input_obj['Sex'])
    sample_pred = model.predict(input_obj)

    return sample_pred