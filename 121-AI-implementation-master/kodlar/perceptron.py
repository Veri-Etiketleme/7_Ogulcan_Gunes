import csv,math,operator
from numpy import random
def loadCSV(path):
	file=open(path)
	csv_content=csv.reader(file)
	csv_content=list(csv_content)
	return csv_content[1:len(csv_content)]
	
def calculateFeatureDotProductAndPredict(county,weights):
		#normalizeCountyData(county)
		dot_prod=weights[0]
		for columnIndex in range (1,len(county)):
			dot_prod+=float(county[columnIndex])*float(weights[columnIndex-1])	
		return 1.0 if dot_prod >= 0.0 else 0.0 

def normalizeCountyData(county):
	totalUSpopulation=325115396
	if(float(county[1])>1):
		county[1]=float(float(county[1])/totalUSpopulation)
		county[2]=float((float(county[2])*float(county[1])/100)/totalUSpopulation)
		county[3]=float((float(county[3])*float(county[1])/100)/totalUSpopulation)
		county[4]=float((float(county[4])*float(county[1])/100)/totalUSpopulation)
		county[5]=float((float(county[5])*float(county[1])/100)/totalUSpopulation)
		county[6]=float((float(county[6])*float(county[1])/100)/totalUSpopulation)
		county[7]=float(float(county[7])*float(county[1])/totalUSpopulation)
		county[8]=float((float(county[8])*float(county[1])/100)/totalUSpopulation)
		county[9]=float(float(county[9])*float(county[1])/totalUSpopulation)
			
def trainPerceptron(trainingset):
	w = [0.0 for i in range(0,10)]
	eta = 0.2
	iterations=500
	for i in range(0,iterations):
		for county in trainingset:
			print('\ncurrent weight -'+str(w))
			print('\nTraining county - '+repr(county))
			result=calculateFeatureDotProductAndPredict(county,w)
			error=float(int(county[0])-result)
			print('\n error obtained - '+str(error))
			w[0] = w[0] + eta * error
			for j in range (1,10):
				w[j]+=eta*error*float(county[j])
	return w	
	
def classifyByPerceptron(testCounty,w):
	result=calculateFeatureDotProductAndPredict(testCounty,w)
	if(result==0):
		print("\nThis county has voted for republician")
		return 0
	else:
		print("\nThis county has voted for democrat")
		return 1


def predictCounty(testset,w):
	predictedVotes=[]
	for county in testset:
		predictedVotes.append(classifyByPerceptron(county,w))
	return predictedVotes
	
def checkAccuracy(testSet,predictedVotes):
	accurateResult=0
	for index in range(0,len(testSet)):
		if(int(testset[index][0])==int(predictedVotes[index])):
			accurateResult+=1
	return (float(accurateResult/len(testSet))*100)

trainingset=loadCSV('votes-train.csv')
testset=loadCSV('votes-test.csv')
weight=trainPerceptron(trainingset)
predictedVotes=predictCounty(testset,weight)
accuracy=checkAccuracy(testset,predictedVotes)
print("\n Overall Prediction accuracy is "+str(accuracy)+"%")
