import numpy as np
import matplotlib.pyplot as plt
from NextWordPredict import LSTM
from DownloadLyrics import GeniusAPI
from SkipGram import *

#----------------------------------------------------------------------------------------------------------------------------------------

# USER INPUT

# These are the 8 metaparameters for the algorithm that can be optimized to improve results
# Everything aside from these is not able to be altered by the user

# Artist's name:
artist = 'Mitski'

# Text file to write lyrics to
txt_file = 'lyricsv1.txt'

# Learning rate of SkipGram
skipLearn = 0.01

# Size of projection layer of SkipGram hidden layer
skipProj = 1000

# Number of strings to write out
num_strings = 20

# Number of words to include in the randomizing set when choosing
per_random = 20

# Length of desired epoch, or "verse"
epoch_length = 30

# Maximum value of the randomly initialized LSTM matrices
rand_mult = 0.1

# Number of nearest words on either side to account for in skipgram
skip_nearest_words = 5

# Maximum value of the randomly initialized SkipGram matrices
skip_mult = 0.1

#--------------------------------------------------------------------------------------------------------------------------------------


skipOptions = {'learning_rate': skipLearn, 'projection_size': skipProj, 'rand_mult': skip_mult}


api = GeniusAPI(artist)
api.download_to_txt(txt_file)

print("Finished Downloading Lyrics")

skip = SkipGram(txt_file, skipOptions)
word_vecs, skipCost = skip.train_to_vectors()

plt.figure
plt.plot(skipCost)
plt.savefig('skipCost.png')

print("Finished Generating Word Vectors")

NWP = LSTM(epoch_length, per_random, rand_mult)
NWPin = NWP.next_word_predict(skip.dictionary, word_vecs)

plt.figure
plt.plot(NWPin['cost'])
plt.savefig('costLSTM.png')

output_predictions = NWP.validate_strings(word_vecs, NWPin['W'], NWPin['U'], NWPin['B'], num_strings)

with open('outputStrings.txt', 'w') as f:
    for row in output_predictions:
        f.write("%s\n\n" %row)
