import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#read in the datasets
cfDict = pd.read_csv('county_facts_dictionary.csv')
totalCountyFacts = pd.read_csv('county_facts.csv')
primaryResults = pd.read_csv('primary_results.csv')

#determine all of the candidates in the race
candidates = []
for temp in primaryResults.candidate:
	if None in candidates:
		candidates.append(temp)
	elif temp not in candidates:
		candidates.append(temp)


winningParty = []
winningCand = []
winningVotes = []
winningFrac = []
fips = []

#Populate a dataFrame of winning info from each county within totalCountyFacts. Since there are mutliple entries per county
#the highest percent of the vote is selected
for index in totalCountyFacts.fips:
	if index in primaryResults.fips:
		value = primaryResults.loc[primaryResults.fips == index]
		if value.empty == False:
			fips.append(index)

			fracVote = value.fraction_votes
			maxId = fracVote.idxmax()
			
			winningParty.append(primaryResults.party.loc[primaryResults.index == maxId].values[0])			
			winningCand.append(primaryResults.candidate.loc[primaryResults.index == maxId].values[0])
			winningVotes.append(primaryResults.votes.loc[primaryResults.index == maxId].values[0])			
			winningFrac.append(primaryResults.fraction_votes.loc[primaryResults.index == maxId].values[0])			

temp = {'fips':fips,'party':winningParty,'candidate':winningCand,'votes':winningVotes,'fraction_votes':winningFrac}

train = pd.DataFrame(temp).set_index('fips')
countyFacts = totalCountyFacts.loc[totalCountyFacts.fips.isin(fips) == True].set_index('fips')
countyFacts = countyFacts.drop(['area_name','state_abbreviation'],axis=1)

#Define the data to train on
train = pd.concat([countyFacts,train],axis=1)

x = train.ix[:,:51]
y = train.ix[:,51:]

#Add a column for the numerical values of both the party and candidates
partyLbl = LabelEncoder()
label = partyLbl.fit_transform(y.party)
y['party_label'] = label

candLbl = LabelEncoder()
label = candLbl.fit_transform(y.candidate)
y['candidate_label'] = label

#split the data into cross validation and a test set. This is used for training the Random Forest
#x_train,x_split,y_train,y_split = train_test_split(x,y,test_size=0.25)
#x_cv,x_test,y_cv,y_test = train_test_split(x,y,test_size=0.5)

x_train = x
y_train = y

#Scale the data
scale = StandardScaler()
scale.fit(x_train)
x_temp = scale.transform(x_train)
x_train = pd.DataFrame(x_temp,index=x_train.index, columns=x_train.columns)

#x_temp = scale.transform(x_cv)
#x_cv = pd.DataFrame(x_temp,index=x_cv.index, columns=x_cv.columns)

#x_temp = scale.transform(x_cv)
#x_test = pd.DataFrame(x_temp,index=x_test.index, columns=x_test.columns)

#Use a random forest to predict the party affiliation
partyClf = RandomForestClassifier(max_depth=20, n_estimators = 30)
partyClf.fit(x_train,y_train.party_label)
#print partyClf.predict_proba(x_cv)

#Predict candidate
candClf = RandomForestClassifier(n_estimators=40, max_depth = 12)
candClf.fit(x_train,y_train.candidate_label)

#Find the counties that appear in the countyFacts but not in the primaryResults
allCountyFacts = totalCountyFacts.set_index('fips')
allCountyFacts = allCountyFacts.drop(['area_name','state_abbreviation'],axis=1)
allPrimaryResults = primaryResults.set_index('fips')

missingFips = []
for fips in allCountyFacts.index:
	if fips not in primaryResults.fips:
		missingFips.append(fips)


missingFacts = allCountyFacts.ix[missingFips]
missingPartyLbl =  partyClf.predict(missingFacts)
missingCandLbl = candClf.predict(missingFacts)
missingParty = partyLbl.classes_[missingPartyLbl]
missingCand = candLbl.classes_[missingCandLbl]

temp = {'fips':missingFips,'candidate':missingCand, 'candidate_label': missingCandLbl, 'party':missingParty, 'party_label': missingPartyLbl}
missingInfo = pd.DataFrame(temp).set_index('fips')

y = pd.concat([y,missingInfo])

#Sort candidates into parties
democrats = ['Hillary Clinton', 'Bernie Sanders', "Martin O'Malley"]
republicans = [x for x in candidates if x not in democrats]
republicans.remove(' Uncommitted')
republicans.remove(' No Preference')

#This step is a bit obscure. Since I know the party of each candidate, if the party and the candidate do not match up after running the prediction model on them then 
#I use the probability given by each classifier to determine if the party is correctly alligned. I've chosen a metric that changes the party if the candidates probability
#is greater than 40% and the probability of the party is less than 60%. This cuts the misalligned parties in more than half, and hopefully reduces the misclassifcation error 
for i in y.index:
	candidate = y.candidate.loc[y.index == i].iloc[0]
	party = y.party.loc[y.index == i].iloc[0]

	maxPartyProb = partyClf.predict_proba(allCountyFacts.loc[allCountyFacts.index == i]).max()
	maxCandProb = candClf.predict_proba(allCountyFacts.loc[allCountyFacts.index == i]).max()
	
	if ((candidate in democrats) and (party != 'Democrat')) or ((candidate in republicans) and (party != 'Republican')):
		if maxPartyProb <= 0.6 and maxCandProb >= 0.4:
			if party == 'Democrat':
				y.set_value(i,'party','Republican')
				y.set_value(i,'party_label',list(partyLbl.classes_).index('Republican'))
			elif party == 'Republican':
				y.set_value(i,'party','Democrat')
				y.set_value(i,'party_label',list(partyLbl.classes_).index('Democrat'))

#Print out the predictions for the counties with the missing primaryResults
y.to_csv('Predictions.csv')
