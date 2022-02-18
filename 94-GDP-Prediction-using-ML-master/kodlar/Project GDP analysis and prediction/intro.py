import pandas as pd                
import matplotlib.pyplot as plt    

def intro():
    print(" \n ------------------------ WELCOME ! ----------------------")
    print(" \nBelow is the List of Countries available in the database ")
    print(" \nThe formula for GDP is: GDP = C + I + G + (Ex - Im),")
    print(" \nwhere C equals household consumption, I equals capital investment, G equals government spending and (Ex - Im) equals net exports,")
    print("\n")
  
def get_code(fn):
    data2 = pd.read_csv(fn, encoding = "ISO-8859-1")
    #print(data2)
    d=list(data2['CID'].unique())      
    e=list(data2['Country'].unique())  
    f=list(zip(d,e))                   
    print(f)

def plotlib(fn,key):
    data2 = pd.read_csv(fn, encoding = "ISO-8859-1")        
    data1=data2[data2['CID'] == key]                        
    ID = data1['CID']                                                        
    cty=data1['Country'].unique()                           
    gdp = data1['GDP']                                      
    con = data1['CON']                                      
    cap = data1['CAP']                                      
    exp = data1['EXP']                                      
    imp = data1['IMP']                                      
    gov = data1['GOV']                                      
    y =data1['Year']                                        
    plt.figure(1)
    plt.ylabel('All KPIs')                  
    plt.xlabel('Year')                      
    plt.title(cty)                          
    plt.plot(y,gdp, color = 'red', label = 'GDP')       
    plt.plot(y,con, color = 'blue', label = 'CON')      
    plt.plot(y,cap, color = 'yellow', label = 'CAP')    
    plt.plot(y,exp, color = 'cyan', label = 'EXP')      
    plt.plot(y,imp, color = 'magenta', label = 'IMP')   
    plt.plot(y,gov, color = 'black', label = 'GOV')     
    plt.legend()
    print(" The generated plot that is displayed is also stored in the path the project is running!")
    plt.savefig(str(cty)+".pdf", format='pdf')          
    plt.show()                                          
