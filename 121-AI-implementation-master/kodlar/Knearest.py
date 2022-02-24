import csv,math,operator
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
	
def classifyByKnearestNeighbour(testCounty,trainingset,k):
	distances=[]
	for county in trainingset:
		distances.append((county[0],calculateEuclideanDistance(county,testCounty)))
		print("comparing counties - \n training county -"+repr(county))
		print("\n test county -"+repr(testCounty))
		distances.sort(key=operator.itemgetter(1))
	neighbors=[]
	for index in range(k):
		neighbors.append(distances[index][0])
	print("Neighbors - \n"+repr(neighbors))
	return classifyCounty(neighbors)

def classifyCounty(neighbors):
	candiates={'republic':0,'democrat':0}
	for vote in neighbors:
		if(int(vote)==0):
			candiates['republic']+=1
		else:
			candiates['democrat']+=1
	if(candiates['republic']>candiates['democrat']):
		print("\nThis county is classified as republic supporter with nearest neighbors = "+str(candiates['republic']))
		return 0
	else:
		print("\nThis county is classified as democrat supporter with nearest neighbors = "+str(candiates['democrat']))
		return 1

def predictCounty(trainingset,testset):
	predictedVotes=[]
	for county in testset:
		predictedVotes.append(classifyByKnearestNeighbour(county,trainingset,5))
	return predictedVotes
	
def checkAccuracy(testSet,predictedVotes):
	accurateResult=0
	for index in range(0,len(testSet)):
		if(int(testset[index][0])==int(predictedVotes[index])):
			accurateResult+=1
	return (float(accurateResult/len(testSet))*100)

trainingset=loadCSV('votes-train.csv')
testset=loadCSV('votes-test.csv')
predictedVotes=predictCounty(trainingset,testset)
accuracy=checkAccuracy(testset,predictedVotes)
print("\n Overall Prediction accuracy is "+str(accuracy)+"%")
