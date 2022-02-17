# Kaggle LANL Earthquake Prediction

This project is my personal solution to this Kaggle competition, it contains 2 main notebooks : 
Feature engineering 
Modeling and Prediction.

## Feature Engineering 
In the Feature engineering part the idea is to extract the most interesting features from a continuous signal. 
I have separated this continuous signal into many segments constituting the learning individuals of our dataset.
I decided to largely limit the number of features describing each individual in order to limit the overfitting. 
The selection of the features was therefore a determining element of my solution.

## Modeling
In the Modeling part I focused on a Gradient Boosting model: LightGBM. 
I brought generalization to my solution by differentiating the learning objectives of the different models and by simply combining their predictions.
