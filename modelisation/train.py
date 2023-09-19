import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
# Data loading and pre-processing
data = pd.read_csv('./data/titanic/train.csv')


def preprocessing(numeric_features, categorical_features):
    """
    Making a preprocessor transformer
    :param dataset: numerical and categorical features to consider
    :return: preprocessor transformer to put in a pipeline
    """
    # Numerical features encoding
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())
               ]
    )
    # Categorical features encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imuter", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
            #("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def trainRFC(X, y):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf


def testing(clf, X, y):
    y_hat = clf.predict(X)
    accuracy = accuracy_score(y_true=y, y_pred=y_hat)
    recall = recall_score(y_true=y, y_pred=y_hat)
    print("Model Accuracy : ", accuracy)
    print("Model recall : ", recall)
    return 0


if __name__ == "__main__":
    # Data loading and pre-processing
    data = pd.read_csv('data/titanic/train.csv')

    delate_features = ['Cabin', 'Name', 'PassengerId', 'Ticket']
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ["Embarked", "Sex", "Pclass"]


    y = data['Survived']
    X = data.drop(['Survived'], axis=1)
    print(X.shape)
    X = X.drop(delate_features, axis=1)
    print(X.shape)
    preprocessor = preprocessing(categorical_features=categorical_features,
                                 numeric_features=numeric_features,
                                 )
    # Model training and testing
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf.fit(X_train, y_train)

    print("model score: %.3f" % clf.score(X_test, y_test))