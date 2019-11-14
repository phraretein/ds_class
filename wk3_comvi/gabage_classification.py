## get data
import pandas as pd
import numpy as np

def getdata(pickleFile):
	dat = pd.read_pickle(pickleFile)
	dat = dat.dropna()
	return dat

## clean data

## preprocess data
def split_data_and_label(dat):
	label = dat['intent'].values.astype('str')
	feat = np.vstack(dat['imgfeat'])
	return label,feat

def feature_extraction(featMat):
	return np.vstack(featMat)


from sklearn.svm import LinearSVC
def trainclassifer(feat,label):
	mod = LinearSVC()
	mod.fit(feat,label)
	return mod

def evaluation_accuracy(prediction,actual):
	acc = sum(prediction==actual)/len(prediction)
	return acc

from sklearn.metrics import confusion_matrix

def pipeline():
	pickleFile = 'train.p'
	tldat = getdata(pickleFile)
	#return tldat
	trainLabel,trainDat = split_data_and_label(tldat)
	#return trainDat
	trainFeat = feature_extraction(trainDat)

	pickleFile = 'test.p'
	tldat = getdata(pickleFile)
	testLabel,testDat = split_data_and_label(tldat)
	testFeat = feature_extraction(testDat)

	mod = trainclassifer(trainFeat,trainLabel)
	res = mod.predict(testFeat)
	lab = list(set(testLabel))
	cm = confusion_matrix(testLabel,res,labels=lab)
	resmat = pd.DataFrame(cm,columns=lab)
	resmat.index=lab
	#return resmat
	
	return evaluation_accuracy(res,testLabel),resmat



