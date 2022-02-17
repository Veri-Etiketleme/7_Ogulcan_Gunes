Instructions to execute the code:

1. The zip file contains a folder named "Data" which contains all the data files. 

2. In our project we have tried classificationn using both imbalanced (original) and balanced (upsampled) dataset. 

3. Without_Upsampling.py performs classification using unbalanced dataset. The required training and testing datasets can be found in found in the "data" folder with names Train.NoEmployeeNumber.csv and Test.NoEmployeeNumber.csv

4. With_Upsampling.py performs classification using balanced dataset. The required training and testing datasets can be found in found in the "data" folder with names upsampledSMOTEnew.csv and Test.NoEmployeeNumber.csv

5. Without_Upsampling.py uses the prediction made from the Neural Nets. It imports a file named Neural_Nets_Prediction_Imbalanced for this.

6. With_Upsampling.py uses the prediction made from the Neural Nets for ensemble learning. It imports a file named Neural_Nets_Prediction_Balanced for this.

7. There are four separate files for neural nets.
