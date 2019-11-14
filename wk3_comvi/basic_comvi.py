import pandas as pd
import numpy as np

## get problem

## get data
def getdata(pickleFile):
	dat = pd.read_pickle(pickleFile)
	return dat

## clean data

## feature extraction
def split(datFrame):
	label = datFrame['intent'].values.astype('str')
	feat = np.vstack(datFrame['imgfeat'])
	return label,feat

def feature_extraction(featMat):
	return np.vstack(featMat)

## model train
from sklearn.svm import LinearSVC
def trainmodel(feat,label):
	modl = LinearSVC()
	modl.fit(feat, label)
	return modl 

## model evaluate
def evaluate(prediction, actual):
	acc = sum(prediction==actual)/len(prediction)
	return acc

## score

def pipeline():
	pickleFile = 'train.p'
	tldat = getdata(pickleFile)
	#return tldat
	trainLabel,trainDat = split(tldat)
	#return trainDat
	trainFeat = feature_extraction(trainDat)

	pickleFile = 'test.p'
	tldat = getdata(pickleFile)
	testLabel,testDat = split(tldat)
	testFeat = feature_extraction(testDat)

	mod = trainmodel(trainFeat,trainLabel)
	res = mod.predict(testFeat)
	
	return evaluate(res,testLabel)