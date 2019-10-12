## get data
import pandas as pd
def getdata(csvFile):
	dat = pd.read_csv(csvFile)
	return dat

## clean data

## preprocess data
def split_data_and_label(dat):
	label = dat['Survived'].values
	feat = dat.drop('Survived',axis=1)
	return label,feat

#def datacleaning(featMat):
#	featMat = featMat.fillna(0)
#	return featMat


## feature extraction
def feature_extraction(featMat):
	#featMat = detectticket(featMat)
	
	featMat = detectnameinit(featMat)
	featMat = detectcabinmissing(featMat)
	#featMat = detectagemissing(featMat)
	featMat = createagerange(featMat)
	featMat = detectkid(featMat)
	featMat = detectsenior(featMat)
	#return featMat
	#featMat['CabinIn'] = featMat['Cabin'].str[0:1]
	#return featMat
	dm = pd.get_dummies(featMat[['Sex','Embarked']])
	featMat = pd.concat([featMat,dm],axis=1)
	#return featMat
	#featMat['namelen'] = featMat['Name'].str.len()
	featMat = featMat._get_numeric_data()
	featMat = featMat.fillna(0)
	##try:
	##	featMat = featMat.drop('CabinIn_T',axis=1)
	##except:
	##	pass
	#featMat = normalize(featMat)
	return featMat

def feature_extraction(featMat):
	featMat = featMat._get_numeric_data()
	featMat = featMat.fillna(0)
	return featMat






from sklearn.ensemble import GradientBoostingClassifier
def trainclassifer(feat,label):
	mod = GradientBoostingClassifier()
	mod.fit(feat,label)
	return mod

def evaluation_accuracy(prediction,actual):
	acc = sum(prediction==actual)/len(prediction)
	return acc

def detectfamilyname(dat):
	dat['lastname'] = [name[0] for name in dat['Name'].str.split(',')]
	namecount = pd.DataFrame(dat.groupby(by='lastname').count()['Name'])
	namecount.columns = ['famcount']
	namecount['lastname'] = namecount.index
	dat = dat.merge(namecount,how='left')
	return dat

def detectagemissing(dat):
	mage = dat['Age'].fillna(-1)
	dat['missingage'] = (mage == -1).astype('int')
	return dat

def detectcabinmissing(dat):
	mcab = dat['Cabin'].fillna(-1)
	dat['missingcabin'] = (mcab == -1).astype('int')
	return dat

def detectkid(dat):
	dat['Kid'] = (dat['Age'] < 12).astype('int')
	return dat
def detectsenior(dat):
	dat['Senior'] = (dat['Age'] > 60).astype('int')
	return dat

def createagerange(dat):
	arList = list(range(0,100,5))
	for i in range(len(arList)-1):
		lb = arList[i]
		ub = arList[i+1]
		dat['ar_'+str(i)] = ((dat['Age']>lb) & (dat['Age']<=ub)).astype('int')
	return dat


def detectticket(dat):
	dat['TicketInit'] = dat['Ticket'].str[0:3]
	return dat

def detectnameinit(dat):
	#dat['NameInit'] = dat['Name'].str[0:3]
	dn = dat['Name'].str.split(',')
	ninit = [n[1][1:3] for n in dn]
	dat['NameInit'] = ninit
	return dat


def pipeline():
	csvFile = 'train.csv'
	tldat = getdata(csvFile)
	trainLabel,trainDat = split_data_and_label(tldat)
	trainFeat = feature_extraction(trainDat)

	csvFile = 'test.csv'
	tldat = getdata(csvFile)
	testLabel,testDat = split_data_and_label(tldat)
	testFeat = feature_extraction(testDat)
	
	mod = trainclassifer(trainFeat,trainLabel)
	
	res = mod.predict(testFeat)
	
	return evaluation_accuracy(res,testLabel)
