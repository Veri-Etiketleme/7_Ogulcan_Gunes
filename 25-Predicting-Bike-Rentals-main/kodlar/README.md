# Predicting Bike Rentals
This project uses a variety of machine learning (ML) methods to predict bike rentals and compare which ML method is the most accurate predictor.

This project uses bike rental data for the city of Washington, D.C. to create machine learning algorithms to predict the total number of bikes people rent in a given hour. We use the following 3 machine learning methods to predict bike rentals: multiple linear regression, decision trees and random forests. We measury the accuracy of each model using a combined score evaluating the root mean squared error (RMSE) and R-Squared (R2) value of each model in the training and test data set and the difference between RMSE score in training and test data sets (the latter 2 metrics evaluate the extent of overfitting in the model which is particularly common for decision trees). 

This project also implements parameter optimization to find the optimal number of minimum leaves and maximum depth for the decision trees and random forest algorithms (with all other parameters using their default value). All 3 final models are compared and evaluated using our error metric. 

This project finds that the optimal machine learning method to predict bike rentals in the city of Washington, D.C. is to use a random forest algorithm with optimal parameters of 2 minimum sample leaves and no maximum depth (using a default number of trees (estimators) of 100).   
