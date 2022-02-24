import csv,math,operator
from numpy import random
import queue 

def loadCSV(path):
	file=open(path)
	csv_content=csv.reader(file)
	csv_content=list(csv_content)
	return csv_content[1:len(csv_content)]
	
def calculateEuclideanDistance(county1,county2):
	if(len(county1)!=len(county2)):
		return sys.maxint
	else:
		distance=0
		normalizeCountyData(county1,county2)
		for columnIndex in range (1,len(county1)):
			distance+=pow(float(county1[columnIndex])-float(county2[columnIndex]),2)
		return math.sqrt(distance)
		
def calculateMeanFeature(county):
	distance=0
	normalizeCountyData(county,county)
	for i in range (1,len(county)):
			distance+=pow(float(county[i]),2)
	return math.sqrt(distance)
		
def normalizeCountyData(county1,county2):
	totalUSpopulation=325115396
	if(float(county1[1])>1):
		county1[1]=float(float(county1[1])/totalUSpopulation)
		county1[2]=float((float(county1[2])*float(county1[1])/100)/totalUSpopulation)
		county1[3]=float((float(county1[3])*float(county1[1])/100)/totalUSpopulation)
		county1[4]=float((float(county1[4])*float(county1[1])/100)/totalUSpopulation)
		county1[5]=float((float(county1[5])*float(county1[1])/100)/totalUSpopulation)
		county1[6]=float((float(county1[6])*float(county1[1])/100)/totalUSpopulation)
		county1[7]=float(float(county1[7])*float(county1[1])/totalUSpopulation)
		county1[8]=float((float(county1[8])*float(county1[1])/100)/totalUSpopulation)
		county1[9]=float(float(county1[9])*float(county1[1])/totalUSpopulation)
	if(float(county2[1])>1):
		county2[1]=float(float(county2[1])/totalUSpopulation)
		county2[2]=float((float(county2[2])*float(county2[1])/100)/totalUSpopulation)
		county2[3]=float((float(county2[3])*float(county2[1])/100)/totalUSpopulation)
		county2[4]=float((float(county2[4])*float(county2[1])/100)/totalUSpopulation)
		county2[5]=float((float(county2[5])*float(county2[1])/100)/totalUSpopulation)
		county2[6]=float((float(county2[6])*float(county2[1])/100)/totalUSpopulation)
		county2[7]=float(float(county2[7])*float(county2[1])/totalUSpopulation)
		county2[8]=float((float(county2[8])*float(county2[1])/100)/totalUSpopulation)
		county2[9]=float(float(county2[9])*float(county2[1])/totalUSpopulation)
			
def generateClusters(trainingset):
	clusters1=queue.PriorityQueue()
	clusters2=queue.PriorityQueue()
	for i in range(0,len(trainingset)):
		for j in range(i+1,len(trainingset)):
			if(int(trainingset[i][0])==0 and int(trainingset[j][0])==0):
				distance=calculateEuclideanDistance(trainingset[i],trainingset[j])
				clusters1.put(distance)
			if(int(trainingset[i][0])==1 and int(trainingset[j][0])==1):
				distance=calculateEuclideanDistance(trainingset[i],trainingset[j])
				clusters2.put(distance)
	while(clusters1.qsize()!=1):
		cluster1=clusters1.get()
		cluster2=clusters1.get()
		meanDistance=float((cluster1+cluster2)/2)
		clusters1.put(meanDistance)
		
	while(clusters2.qsize()!=1):
		cluster1=clusters2.get()
		cluster2=clusters2.get()
		meanDistance=float((cluster1+cluster2)/2)
		clusters2.put(meanDistance)
	
	clusters=[clusters1.get(),clusters2.get()]
	return clusters

	

def predictCounty(testset,clusters):
	predictedVotes=[]
	centroids=clusters
	result=0
	for county in testset:
		print("\nTest county -"+repr(county))
		distanceFromCluster1=math.sqrt(pow(calculateMeanFeature(county)-centroids[0],2))
		distanceFromCluster2=math.sqrt(pow(calculateMeanFeature(county)-centroids[1],2))
		if(distanceFromCluster1<distanceFromCluster2):
			print("\n This is county has been predicted as supporter of Republician")
			predictedVotes.append(0)
		else:
			print("\n This is county has been predicted as supporter of Democrat")
			predictedVotes.append(1)
	return predictedVotes
	
def checkAccuracy(testSet,predictedVotes):
	accurateResult=0
	for index in range(0,len(testSet)):
		if(int(testset[index][0])==int(predictedVotes[index])):
			accurateResult+=1
	return (float(accurateResult/len(testSet))*100)

trainingset=loadCSV('votes-train.csv')
testset=loadCSV('votes-test.csv')
clusters=generateClusters(trainingset)
predictedVotes=predictCounty(testset,clusters)
accuracy=checkAccuracy(testset,predictedVotes)
print("\n Overall Prediction accuracy is "+str(accuracy)+"%")
