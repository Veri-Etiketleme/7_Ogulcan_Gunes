#reference is taken from http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html

import csv,math,operator
from numpy import random
import queue 

def loadCSV(path):
	file=open(path)
	csv_content=csv.reader(file)
	csv_content=list(csv_content)
	return csv_content[1:len(csv_content)]

def divideset(rows,column,value):
   split_function=None
   if isinstance(value,int) or isinstance(value,float): 
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)

def uniquecounts(rows):
   results={}
   for row in rows:
      r=row[0]
      if r not in results: results[r]=0
      results[r]+=1
   return results
 

def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)  
   results=uniquecounts(rows)
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent

class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

def buildtree(rows,scoref=entropy): 
  if len(rows)==0: return decisionnode() 
  current_score=scoref(rows)

  best_gain=0.0
  best_criteria=None
  best_sets=None
  
  column_count=len(rows[0])	  
  for col in range(1,column_count):
    global column_values     
    column_values={}            
    for row in rows:
       column_values[row[col]]=1   
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value) 
      p=float(len(set1))/len(rows) 
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) 
      if gain>best_gain and len(set1)>0 and len(set2)>0: 
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
        
  if best_gain>0:
    trueBranch=buildtree(best_sets[0])
    falseBranch=buildtree(best_sets[1])
    return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisionnode(results=uniquecounts(rows))
	

def printtree(tree,indent=''):
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col)+':'+str(tree.value)+'? ')
        print(indent+'T->', end=" ")
        printtree(tree.tb,indent+'  ')
        print(indent+'F->', end=" ")
        printtree(tree.fb,indent+'  ')



def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)
	

def predictCounty(testset,tree):
	predictedVotes=[]
	for county in testset:
		print("\nTest county -"+repr(county))
		results=classify(county,tree)
		
		for key in results.keys():
			classifiedData=key
		if(int(classifiedData)==0):
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
tree=buildtree(trainingset)
printtree(tree)

predictedVotes=predictCounty(testset,tree)
accuracy=checkAccuracy(testset,predictedVotes)
print("\n Overall Prediction accuracy is "+str(accuracy)+"%")
