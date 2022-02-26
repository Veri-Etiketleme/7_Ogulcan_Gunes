# lyric-next-word-prediction
Next word prediction model which uses a series of neural networks and API requests to train a next word prediction model based on the lyrics from a user input artist

-----------------------------------------------------------------------------------------------------------------------------------

## User Notes:

  * DownloadLyrics.py contains the Genius API
  * SkipGram.py obviously contains the word encoding
  * NextWordPredict.py contains the LSTM and dependencies
  * LNWP.py contains the master file that pulls and executes the classes from the files
  
 Python Modules Required
  * requests
  * beautifulsoup4
  * urllib
  * numpy
  * matplotlib
  * csv

-----------------------------------------------------------------------------------------------------------------------------------

## Program contains three parts:

  1. Genius API
  2. SkipGram Word Encoding
  3. LSTM training for Next Word Prediction
  
-----------------------------------------------------------------------------------------------------------------------------------
  
## Part One: API

  Overview: Genius API takes an input artist and crawls Genius.com for all of their song lyrics, then downloads them to a text file
  
  1. Searches Genius for the artist and their corresponding ID [6]
  2. Stores all of the artist's song URLs   [5]
  3. Reads the lyrics from the HTML embedding from each song URL
  4. Saves the output to a text file
  
-----------------------------------------------------------------------------------------------------------------------------------
  
## Part Two: SkipGram

  Overview: Cleans and parses the raw lyrics into an array, then trains a neural network to encode words as vectors
  
  1. Cleans terms and characters (i.e. '[Pre-Chorus]') from the raw lyrics text file
  2. Parses the words from the text file into an array of strings
  3. Creates a dictionary of all unique words
  4. Uses forward and back propagation to train a SkipGram model [7] [8]
      * Input: Hot-encoded word vectors corresponding to the word's position in the dictionary
      * Output: Word predicted to be most similar to the input word
      * Cost: k words that precede and follow the input word (cross-entropy [9])
  5. Encodes each word by passing it through the fully trained network and generating its output
  6. Returns the original text array and a dictionary with words and their corresponding vector encoding
  
-----------------------------------------------------------------------------------------------------------------------------------

## Part Three: LSTM

  Overview: Trains a time-series LSTM (Variant of recurrent neural network) on artist song lyrics and creates strings based on trained weights
  
  1. Divides raw lyric array into epochs to pass through training
  2. Uses forward and back propagation to train the LSTM network [1-4]
      * Input: vector encoding of words along range(n * i, n * (i+1))
        * n: Size of each epoch (constant)
        * i: Current epoch number (iterated)
      * Output: Pseudo-vector encoding for classification using iterative cross-entropy loss minimization
      * Cost: Cross entropy loss against the next word in the epoch sequence
  3. Builds strings based on the trained weights
      * Generates a single random word from the original dictionary
      * Passes this word into the first time input (No other input is given)
      * The first ouput is parsed as a word by using iterated cross-entropy loss minimization [9]
            i.e. cost for output vector is found against each individual encoded vector in the dictionary
      * This generated word is then passed as the input to the next time series
      * String is formed from all of these words which are passed into one another
  4. Returns all of these strings
  
-----------------------------------------------------------------------------------------------------------------------------------

## Author Info

* Robbie Spiers
* Idaho State University
* Correspondence to spierob2@isu.edu

*All code here was written exclusively by me
  
-----------------------------------------------------------------------------------------------------------------------------------
  
## References:

[1] https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9

[2] https://towardsdatascience.com/back-to-basics-deriving-back-propagation-on-simple-rnn-lstm-feat-aidan-gomez-c7f286ba973d

[3] https://colah.github.io/posts/2015-08-Understanding-LSTMs/

[4] https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/

[5] https://github.com/johnwmillr/LyricsGenius/blob/master/lyricsgenius/api.py

[6] https://genius.com/api-clients

[7] https://medium.com/district-data-labs/forward-propagation-building-a-skip-gram-net-from-the-ground-up-9578814b221

[8] http://www.claudiobellei.com/2018/01/06/backprop-word2vec/

[9] https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
