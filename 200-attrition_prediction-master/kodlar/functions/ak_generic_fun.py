# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:17:39 2019

@author: ashish_kumar2
"""

# This file contains the following generic functions
# 1. Check duplicate values in rows and columns
# 2. Get details of columns with empty values
# 3. Calculate sets of highly correlated data
# 4. Remove highly correlated columns from the passed data
# 5. Extract all information in the datetime column of a dataframe
# 6. Get list of unique values in a column



# Import required packages
import pandas as pd
import numpy as np
import datetime



#----------1. Function to check duplicate values in rows and columns----------#
def dup_col_rows(ds):
    # Check for duplicated columns
    orig_col=len(ds.columns)
    unique_col_ds = ds.loc[:,~ds.columns.duplicated()]
    new_col=len(unique_col_ds.columns)
    if (orig_col==new_col):
        print("No Duplicate columns found")
    else:
        print("Removed duplicate columns")
    
    # Check for duplicated rows
    if(any(pd.DataFrame.duplicated(unique_col_ds)==True)):
        bef_len=len(unique_col_ds)
        unique_ds=pd.DataFrame.drop_duplicates(unique_col_ds,axis=0)
        aft_len=len(unique_ds)
        print("Removed %s Duplicated Rows" %(bef_len-aft_len))
        return(unique_ds)
    else:
        print("No Duplicated rows Found")
        return(unique_col_ds)
#-----------------------------------------------------------------------------#
        
        


        
#---------2. Function to get details of columns with empty values-------------#
def get_null(ds):
    num_data= ds.isnull().sum()
    if (len(num_data[num_data>0])>0):
        null_info=num_data[num_data>0]
        print("Columns have null values, Please refer null_info dataset")
        return(null_info)
    else:
        print("There are no null values in the dataset")
#-----------------------------------------------------------------------------#





#-----------3. Function to calculate sets of highly correlated data-----------#
def high_corr_sets(ds,corr_coef):
    # Set correlation ratio
    corr_coef=0.9
    
    # Get correlation Matrix
    ds_corr=ds.corr().abs()
    
    sets=pd.DataFrame()
    
    
    r_name=list(ds_corr.index)
    c_name=list(list(ds_corr))
    
    corr_u= pd.DataFrame(np.triu(ds_corr))
    
    corr_u.columns=c_name
    corr_u.index=r_name
    
    corr_u=corr_u.astype(float)
    
    #Flatten Correlation Matrix
    corr_flatenned = corr_u.stack().reset_index()
    corr_flatenned.columns=['First_Parameter','Second_Parameter','Correlation']
    
    
    req_set=corr_flatenned[corr_flatenned.Correlation>=corr_coef]
    i=0
    while(len(req_set)>0):
        rep_name=req_set.First_Parameter.iloc[0,]
        temp=req_set[req_set.First_Parameter==rep_name]
        if(len(temp)>1):
            aaa=temp[['Second_Parameter','Correlation']]
            aaa=aaa.sort_values(by='Correlation', ascending= False)
            aaa=aaa.reset_index(drop=True)
            sets=pd.concat([sets,aaa],ignore_index=True,axis=1)
            req_set=req_set.drop(req_set[req_set.First_Parameter.isin(aaa.Second_Parameter)].index.tolist())
            i=i+1
        else:
            req_set=req_set.drop(req_set[req_set.First_Parameter==rep_name].index)
        #print(len(req_set))
        
    sets=sets.fillna('')
    
    if (len(sets)>0):
        print("Correlated sets available as per correlation ratio = %s"%corr_coef)
    
    return sets
#-----------------------------------------------------------------------------#        






#-----4. Function to remove highly correlated columns from the passed data----#
def rem_high_corr(ds, corr_coef):
    # Create correlation matrix
    corr_matrix = ds.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > corr_coef)]
    
    ds=ds.drop(to_drop, axis=1)
    
    return ds        
#-----------------------------------------------------------------------------#         




#-----------------------------------------------------------------------------#        
#-5. Function to extract the below information from datetime column of a pandas#

    # 1. Day
    # 2. Month
    # 3. Year
    # 4. Quater
    # 5. Semester
    # 6. Day of the Week
    # 7. Weekday Name
    # 7. Weekend or not
    # 8. Leap Year or Not
    # 9. Hour
    # 10.Minute
    # 11.Second
    
# The below function will replace the original colums and append the above 
# parameters to the existing dataframe with the user passed prefix
    
def date_time_info_extract (ds,datetime_col,prefix):
    
    # Create a new dataframe to store all extracted inforamtion
    date_info= pd.DataFrame()
    
    # Convert the selected column to datetime format
    date_info[datetime_col]=pd.to_datetime(ds[datetime_col])
    
    # Extract Day from the selected column
    date_info['day']= date_info[datetime_col].dt.day
    
    # Extract month from the selected column
    date_info['month']= date_info[datetime_col].dt.month
    
    # Extract Year form the selected column
    date_info['year']= date_info[datetime_col].dt.year
    
    # Extract Quarter for the selected column
    date_info['quarter'] = date_info[datetime_col].dt.quarter
    
    # Extract semester from the selected column
    # Quarter 1 and 2 are semester 1 and quarter 3 and 4 are in semester 2
    # We will be using where and is in function for the same
    #ds['semester'] = np.where(date_info[datetime_col].dt.quarter.isin([1,2]),1,2)
    date_info['semester'] = np.where(date_info.quarter.isin([1,2]),1,2)
    
    # Extract day of the week from the selected columns
    date_info['dayofweek'] = date_info[datetime_col].dt.dayofweek
    
    ## Extract weekday name frm the selected column
    ## Monday starts with 0
    #date_info['dayofweek_name']=date_info[datetime_col].dt.weekday_name
    
    # Extract weekend or not information from the selected column
    date_info['is_weekend']= np.where(date_info.dayofweek.isin(['6','5']),1,0)
    
    # Extract Leap year information from the selected column
    date_info['is_leap_year']= date_info[datetime_col].dt.is_leap_year
    
    # Extract hour from the selected column
    date_info['hour'] = date_info[datetime_col].dt.hour
    
    # Extract hour from the selected column
    date_info['minute'] = date_info[datetime_col].dt.minute
    
    # Extract hour from the selected column
    date_info['second'] = date_info[datetime_col].dt.second
    
    
    # Drop original column from the  data_info dataset
    date_info=date_info.drop(datetime_col, axis=1)
    
    # Prefix custom text for each extracted columns
    prefix='DOB_'
    date_info = date_info.add_prefix(prefix)
    
    # Drop selected datetime column from the original dataframe
    ds = ds.drop(datetime_col, axis=1)
    
    # Merge both the extracted info with that of original dataset
    ds=pd.concat([ds,date_info], axis=1)
        
    return ds
#-----------------------------------------------------------------------------#
    


#-----------------Get count Unique values in a column-------------------------#
# This observation will be halpful in getting idea for creating dummy values for 
# a variable
def get_cat_count(ds):
    ret_df=pd.DataFrame(columns=['Header','Unique_Count'])
    for i in range(len(ds.columns)):
        curr_col=ds.columns[i]
        temp=pd.DataFrame({"Header":curr_col,'Unique_Count':len(ds[curr_col].value_counts())}, index=[0])
        ret_df=ret_df.append(temp)
    ret_df=ret_df.sort_values(by=['Unique_Count'])
    return ret_df
#-----------------------------------------------------------------------------#