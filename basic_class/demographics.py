# Demographics Prediction
# 01 Oct 2019
# Bhudharhita Teinwan
import pandas as pd
from sklearn.svm import LinearSVC


## get problem

## get data
# weight/height
def getData(excelFile):
    dat = pd.read_excel(excelFile)
    return dat

# feature extraction
def featureExtract(dataFrame):
    label = dataFrame['gender'].values
    feat = dataFrame[['weight', 'height']].values
    return label, feat

# model train
def trainModel(feat,label):
    modl = LinearSVC()
    modl.fit(feat, label)
    return modl
# model evaluate

# score


def runProcess(excelFile):
    dat = getData(excelFile)
    label, feat = featureExtract(dat)
    modl = trainModel(feat, label)
    return modl