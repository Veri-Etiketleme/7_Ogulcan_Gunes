# Next-Word-Prediction-From-Scratch

Simple Next Word Prediction From Scratch, using only Numpy.

Created as a course project for CMPE 597 Deep Learning course.

Example Results:

city   of   new -->  york

life   in   the -->  world

he   is   the -->  best

------------------------------------------------------------------------------------------

Requirements to Run:

data folder should be in the same directory with other files.
It should contain all provided numpy files.
In order to run eval.py and tsne.py, model.pk file should be in the same directory

------------------------------------------------------------------------------------------

Training the Model:

Running the main.py file will start the training.
It is now set up with following settings: learning_rate=0.001, batch_size=500, epochs=35.
Data is divided into batches and shuffled in the training function.
Training takes around 2 hours.
At the end of every epoch, validation accuracy is reported.
At the end of the training, loss curve and curve for validation accuracy are reported.
These curves are included in the file since training takes long time
At the end of training, final test and validation accuracy are reported.

------------------------------------------------------------------------------------------

Evaluation:

Running the eval.py file will load the model (model.pk) and calculate the test accuracy.
It will also report some tests, showing some example guess results.
New words can be checked by using guess_next_word function just by giving words (words should be in the vocab)

-------------------------------------------------------------------------------------------

tSNE:

Running the tsne.py file will load the model (model.pk) and creates a 2-D plot of the embeddings, and show the result.
An example result is included in the file.

-------------------------------------------------------------------------------------------
