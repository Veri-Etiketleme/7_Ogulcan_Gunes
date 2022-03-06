# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:53:31 2019

@author: ashish_kumar2
"""
# This is a plotting function and performs the below tasks
# 1. Plot numeric and categorical variables in the same dataset with the birds eye view
# 2. Plot Correlation matrix for a dataset
# 3. Plot ROC_AUC Curve for classification Algorithm
# 4. Function to get confusion metrices with labels and plot it

# NOTE- ALL PLOTS WILL BE SAVED AT THE PASSED LOCATION


# Code to Plot Numeric and categorical variables present is the same dataframe


import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
# 1. Function to plot numeric and categorical variables present in the same dataset#
# This code was developed during analysis for IBM HR Attrition dataset
# It will generate a single plot for all the numeric data columns and
# another single plot for all the categorical data columns.

def plt_numeric_categorical_cols(ds, out_loc):
        
    # Select numeric columns
    numeric_cols_ds=  list(ds.select_dtypes(include=['floating','integer','integer','datetime64']).columns)
       
    
    # Data Visualization
    # plot all the numeric variables
    numeric_ds=ds[numeric_cols_ds]
    
    numeric_ds.plot(subplots=True,
                    layout=(7,4),
                   figsize=(40,30),
                   legend=False,
                   fontsize=20,
                   title=list(numeric_ds.columns))
    plt.savefig(out_loc+"/"+'All_numeric_Plots.jpeg', bbox_inches='tight' ,dpi=100 )


    # Plot all categorical variables
    # Select Non Numeric columns
    non_numeric_cols= list(ds.select_dtypes(include=['object']).columns)
    
    categorical_ds = ds[non_numeric_cols]
    
    
    # Plot all categorical parameters in a single plot for a birds eye visualization
    nrow=6
    ncol=3
    # make a list of all dataframes 
    df_list = list(categorical_ds.columns)
    #fig= plt.figure(40,30)
    #axes = fig.add_subplots(nrow, ncol)
    fig, axes = plt.subplots(nrow, ncol,figsize=(20,40))
    # plot counter
    count=0
    for r in range(nrow):
        for c in range(ncol):
            #df_list[count].plot(ax=axes[r,c])
            if count>len(df_list):
                plt.text("Test",ax=axes[r,c])
                count=count+1
            else:
                categorical_ds.iloc[:,count].value_counts().sort_index().plot.bar(rot=20,
                                                                       ax=axes[r,c],
                                                                       title=categorical_ds.iloc[:,count].name,
                                                                       fontsize=10)
                count=count+1
    
    plt.savefig(out_loc+"\\"+'All_Categorical_Plots.jpeg',bbox_inches='tight' ,dpi=100 )
    
###############################################################################






###############################################################################
#-------------- 2. Function to plot Correlation matrix for a dataset----------#
def plt_corr_mat_heatmap(ds,out_loc):
    f,ax = plt.subplots(figsize=(50, 50))
    sns.heatmap(ds.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(out_loc+"/"+"Correlation_Matrix",bbox_inches='tight',dpi=200)
###############################################################################    
    
    
    
    


###############################################################################
#------3. Function to Plot ROC_AUC Curve for classification Algorithm---------#
def roc_auc_curve_plot(y_true,y_pred,classifier,test_type,model_name,x_true, out_loc):
    #y_true<- actual dependent variable
    #y_pred<- predicted dependent variable
    #classifier<- name of the trained classifier
    #test_type<- This is a string. "Validation" or "Training"
    #x_true<- this data on wich the prediction is to be done
    #out_loc<- Output location to save plot
    
    # Import Packages
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    

    # Since roc_suc score takes probablities of one of the classes as scores.
    # We need to calculate the scores
    y_prob=classifier.predict_proba(x_true)
    
    # Keep probablities of positive outcome only
    y_prob_yes=y_prob[:,1]
    
    # Now calcualte the roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_true=y_true,y_score=y_prob_yes)
    roc_auc_score_data = auc(fpr, tpr)
    
    # Plot roc_auc_curve for training set
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label='ROC curve (area = %0.2f)' % roc_auc_score_data)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (TPR)')
    plt.ylabel('True Positive Rate (FPR)')
    plt.title('Receiver operating characteristic - '+ test_type+" Set"+' ( '+model_name+' )')
    plt.legend(loc="lower right")
    plt.savefig(out_loc+"\\"+'ROC_'+test_type+'_'+model_name+".jpeg",bbox_inches='tight' ,dpi=100 )
    #plt.show()
###############################################################################
    
    



###############################################################################
#-------4. Function to get confusion metrices with labels and plot it---------#
def label_and_plot_confusion_matrix(y_true,y_pred,test_type,model_name, out_loc):
    
    # Import packages    
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    
    # Get unique class and sort it in descending order
    unique_label=np.unique([y_true,y_pred])
    unique_label=-np.sort(-unique_label)
    
    # Get confusion Matrix
    confusion_matrix_valid=confusion_matrix(y_true,y_pred, labels=unique_label).T
    # Here I am transposing the matrix as its easier for me to interpret the results.
    # This is entirely personal choice
    
    # Add Labels to confusion matrix
    confusion_matrix_valid=pd.DataFrame(confusion_matrix_valid,
                                    index=[x for x in unique_label], 
                                    columns=[x for x in unique_label])
    
    # Plot data using seaborn heatmap
    f,ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(confusion_matrix_valid, annot=True, cbar= False,fmt= '.0f',ax=ax)
    # The below line of code is to resolve a bug in matplotlib, where the 
    # top and bottom labels were cut in half for heatmap
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    # Add Labels and titles
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title('Confusion Matrix - '+test_type+" Set"+' ( '+model_name+' )')
    
    # Save plot
    plt.savefig(out_loc+"/"+"Confusion_Matrix_"+test_type+'_'+model_name+".jpeg",bbox_inches='tight',dpi=200)
    
    # update label of confusion matrix for retutning
    confusion_matrix_valid.index= ['Predicted:{:}'.format(x) for x in unique_label]
    confusion_matrix_valid.columns= ['Actual:{:}'.format(x) for x in unique_label]
    
    return confusion_matrix_valid
###############################################################################