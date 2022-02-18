import pandas as pd                     
from statsmodels.api import OLS         
class file_data:
	def __init__(self, filename, key):              
		self.fn=filename
		self.key=key
	
	# Function to get data
	def get_data(self,key):                         
                X = []
                Y = []
                data2 = pd.read_csv(self.fn, encoding = "ISO-8859-1")
                data1=data2[data2['CID'] == self.key]    
                Y=data1['GDP']                           
                X=data1[['CON','GOV','CAP','EXP','IMP']] 
                return X,Y                               

class OLSModel(file_data):                              
	def __init__(self, filename, X, Y, key):        
		file_data.__init__(self, filename, key)
		self.y=Y
		self.x=X
		self.key=key
		
	def olsm(self):                                 
                model = OLS(self.y, self.x)
                result = model.fit()                    
                return result                           

