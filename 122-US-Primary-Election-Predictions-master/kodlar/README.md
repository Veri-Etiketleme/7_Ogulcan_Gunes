# US-Primary-Election-Predictions

The purpose of this code is to be able to successfully predict both the candidate and the party that each county in the united states voted for in the primary elections based on their population statistics. 
 
I used a random forest algorithm to predict both the candidate and the party associated with each county. I used a cross validation test and determined that these predictions generated scores well into the mid 90's, so the actual error on the test set should be fairly low. I then used an approach to further reduce the error by sorting the candidates with misaligned parties and changing the parties if they met a certain threshold. The results of the analysis are within Predictions.csv
 
The main body of code is within elections.py. It takes in the data sets from county_facts.csv, county_facts_dictionary.csv, and primary_results.csv, structures it, and then comes up with an accurate model to predict both party and candidate for each county. 
 
There's plenty more analysis I could do on this data set in the future. One possibility is taking my predictions and seeing if I can accurately predict the winner of the general election based on the parties that won each county.
