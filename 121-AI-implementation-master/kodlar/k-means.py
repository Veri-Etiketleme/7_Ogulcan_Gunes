import csv,math,operator
from numpy import random

def loadCSV(path):
	file=open(path)
	csv_content=csv.reader(file)
	csv_content=list(csv_content)
	return csv_content[1:len(csv_content)]
	
def calculateFeaturesSD(county):
	normalizeCountyData(county)
	mean=0.0
	for columnIndex in range (1,len(county)):
		mean+=county[columnIndex]
	mean=(mean/(len(county)-1))
	sd=0.0
	for columnIndex in range (1,len(county)):
		sd+=pow(county[columnIndex]-float(mean),2)
	sd=(sd/(len(county)-1))
	return math.sqrt(sd) 

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
			
def generateClusters(trainingset):
	cluster1 = []
	cluster2 = []
	for county in trainingset:
		sd=calculateFeaturesSD(county)
		if(int(county[0])==0):
			cluster1.append(sd)
		else:	
			cluster2.append(sd)
	clusters=[cluster1,cluster2]
	return clusters
	
def calculateEuclideanDistance(data1,data2):
	distance=pow(float(data1)-float(data2),2)
	return math.sqrt(distance)

def getCentroids(clusters):
	cluster1=clusters[0]
	cluster2=clusters[1]
	centroid1=0.0
	centroid2=0.0
	for sd in cluster1:
		centroid1+=sd
	centroid1=centroid1/len(cluster1)
	for sd in cluster2:
		centroid2+=sd
	centroid2=centroid2/len(cluster2)
	centroids=[centroid1,centroid2]
	return centroids
	

def predictCounty(testset,clusters):
	predictedVotes=[]
	centroids=getCentroids(clusters)
	result=0
	for county in testset:
		print("\nTest county -"+repr(county))
		distanceFromCluster1=calculateEuclideanDistance(calculateFeaturesSD(county),centroids[0])
		distanceFromCluster2=calculateEuclideanDistance(calculateFeaturesSD(county),centroids[1])
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
