import pandas as pd

## get problem

## get data
def getdata(excelFile):
	dat = pd.read_excel(excelFile)
	return dat

## clean data

## feature extraction
def featureextract(datFrame):
	label = datFrame['gender'].values
	feat = datFrame[['weight','height']].values
	return label,feat


## model train
from sklearn.ensemble import GradientBoostingClassifier
def trainmodel(feat,label):
	modl = GradientBoostingClassifier()
	modl.fit(feat, label)
	return modl 

## model evaluate

## score

def runprocess(excelFile):
	dat = getdata(excelFile)
	label,feat = featureextract(dat)
	modl = trainmodel(feat,label)

	return modl