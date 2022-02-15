# Kaggle-Titanic

This is my prediction model for Kaggle's Titanic: Machine Learning from Disaster competition.

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

The passengers are divided into training and testing sets and the goal of this challenge was to develope a machine learning model to make accurate predictions on the testing set regarding which passengers survived and which passengers died when the Titanic sank.

I tried many different approaches to this machine learning challenge and finally settled on this model utilizing a Voting Classifier with Gradient Boost, Random Forest, Decision Tree, K Neighbors, and Support Vector Classifier models as my base estimators. This final model was able to achive an accuracy score of 0.80382 when making predicitons on the testing set landing it in the top 8% of scores on the public leaderboard (although it actually would be higher if the top of the leaderboard wasn't full of 100% accuracy illegitimate submissions). To my knowledge, the highest legitimate score that has been achieved is 0.84668 by Chris Deotte with his WCG+XGBoost model.
