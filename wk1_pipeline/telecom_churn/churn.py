
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier


def read_data(file):
    dat = pd.read_csv(file)
    return dat


def clean_data(df):
    # nan_cols = [i for i in train_df.columns if train_df[i].isnull().any()]
    # train_df.isnull().sum()

    # Filling null with the mean
    df['TotalCharges'].fillna((df['TotalCharges'].mean()), inplace = True)
    #df = df.fillna(0)
    return df


def get_feature_label(df):
    label = df['Churn'].values
    feature = df.drop('Churn', axis=1)
    return feature, label


def get_feature(df):
    obj_cols, num_cols = categorize_type(df)
    feature = pd.concat([pd.get_dummies(df[obj_cols]), df[num_cols]], axis=1)
    return feature


def categorize_type(df):
    obj_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'customerID']
    num_cols = [col for col in df.columns if df[col].dtype != 'object']
    return obj_cols, num_cols


def train(feature, label):
    model = GradientBoostingClassifier()
    model.fit(feature, label)
    return model

def evaluation(prediction, actual):
    acc = sum(prediction == actual)/len(prediction)
    return acc

def get_feature_importance(model, X_train, X_test):
    im = model.feature_importances_
    im_index = np.argsort(-im) 
    X_train = X_train.iloc[:,im_index[0:20]]
    X_test = X_test[X_train.columns] # getting the column that are the same as X_train
    return X_train, X_test
    # --- Below only give the name of the importance feature ---
    # feat_ls = [df.columns.values[item] for i, item in enumerate(ls[:stop])]
    # return feat_ls

def pipeline():
    train_path = 'train.csv'
    test_path = 'test.csv'
    test_df = read_data(test_path)
    train_df = read_data(train_path)

    # --- Train data ---
    train_df = clean_data(train_df)
    train_feat, Y_train = get_feature_label(train_df)
    X_train = get_feature(train_feat)
    model = train(X_train, Y_train)


    # --- Test data ---
    test_df = clean_data(test_df)
    test_feat, Y_test = get_feature_label(test_df)
    X_test = get_feature(test_feat)

    # --- Getting importance feature ----
    X_train,X_test = get_feature_importance(model,X_train,X_test)
    model = train(X_train, Y_train)

    # -- Predicting ---
    result = model.predict(X_test)
    
    return evaluation(result, Y_test)

