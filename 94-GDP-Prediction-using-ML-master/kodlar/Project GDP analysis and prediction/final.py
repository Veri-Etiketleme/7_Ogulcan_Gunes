from intro import *            
from filedata import *
def final():
    filename = 'DataFinal.csv'                                          
    intro()                                                         
    get_code(filename)                                              
    key=int(input("\n Please select a Country of your choice:"))     
    if key in range(1,212):                                         
        fd=file_data(filename,key)                                  
        X,Y = fd.get_data(key)                                      
        om=OLSModel(filename,X, Y, key)                             
        res=om.olsm()                                               
        print(res.summary())                                        
        plotlib(filename,key)                                       
    else:
        print(" \n Wrong country code , Check again")               

r = 'Y'                                                             
while (r == 'Y' or r =='y'):                                        
    final()
    r=input("\n Press y if you want to generate analysis for another country: ")

