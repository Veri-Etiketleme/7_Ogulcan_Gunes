#  Spelling Correction 

BBM497 - Introduction to NLP Lab.

Assignment 2

### Author
Ebubekir Yiğit - 21328629



### Project Introduction
First, we read the incorrectly made words and the correct spelling 
by reading the data separated by error tags such as 
```
<ERR targ=word> wordd </ERR>
```
and recorded the mistakes made here. 
We then looked at the edit distance between these mistakes and the words in the data. 
We recorded the words with edit distance value 1. 
Later we recorded the letter changes between these words. 
(The possibility of another word instead of a word)

By using the Hidden Markov Model and the Viterbi Algorithm, 
we compare the words we find with the Error tag to find words 
that can be used instead of misspelled words. 
If we found it right, it's more likely. found/total.
## Getting Started

In the project, libraries are written for Minimum Edit Distance, 
N-Gram Model and Hidden Markov Model. We will use "bigram" and "unigram" models with N-Gram.

When we look at the NLP module, we see the classes of these three libraries.

MinimumEditDistance takes two parameters and gives properties such as edit distance, 
misspell index and correction type between these words.

In the N-Gram model, we can create the desired model with the supplied train set, 
N-Gram and isSmoothed parameters.

HiddenMarkovModel uses the Viterbi algorithm to calculate the probabilities of 
the cue tags with the error tag and gives the ratio of words we know in total.

**All operations are progressively saved to the console.log file. 
You can browse the console.log file to see the operations.**


**Minimum Edit Distance Library:**
```python
    import NLP
    
    edit_distance = NLP.MinimumEditDistance(correct="ebu", wrong="abu").calculate()
    
    transformation_type = edit_distance.getCorrectionType()     # returns "substitution"
    index = edit_distance.getMisspellIndex()                    # returns 0
    distance = edit_distance.getEditDistance()                  # returns 1
    edit_chars = edit_distance.getEditChars()                   # returns (e,a)
```


**N-Gram Model Library:**
```python
    import NLP
    
    unigram_model = NLP.NGramModel(sentences=sentence_list,N_gram=NLP.UNIGRAM).createModel()
    
    frequency_dict = unigram_model.getModelFrequency()         # returns N-Gram frequency dict 
```   

**Hidden Markov Model Library:**
HMM V1 is more efficient than v2.
```python
    import NLP
    
    founded, total = NLP.HiddenMarkovModel(error_line, unigram_model, bigram_model,
                                                              misspelling_chars_hash, ins_del_subs_hash, output_file) \
                        .generateHmmModel() \
                        .getFoundPerTotal()
    result = round((total_found / total_error) * 100)
```   

**Hidden Markov Model Library v2:**

The algorithm of HMM V2 is better than v1.
```python
    import NLP_v2
    
    founded, total = NLP_v2.HiddenMarkovModel(error_line, unigram_model, bigram_model,
                                                              misspelling_chars_hash, ins_del_subs_hash, output_file) \
                        .generateHmmModel() \
                        .getFoundPerTotal()
    result = round((total_found / total_error) * 100)
```   

**Default HMM version is v1 but you can change it from constants.py file.
(HMM_VERSION = 2)**


### Output

**For HMM V1:**
```  
###################################     Evaluation Result: %42     ###################################
```  


**For HMM V2:**
```  
###################################     Evaluation Result: %36     ###################################
``` 


### Prerequisites

Python 3.5 is required to run the project. 

**Used Python libraries:**

- numpy


For Ubuntu:

```
sudo apt-get install python3.5
```

For Centos:
```
sudo yum -y install python35u
```

### Running the Project

python3 assignment2.py <text_dataset> <output_file>

Example:
```
python3 assignment2.py dataset.txt output.txt
```

**After run, please see "console.log" file to show process logs.**

### console.log File
```
22:13:51   LOG/ MAIN-PROCESS:  Spelling Correction program is started.
22:13:51   LOG/ VERSION:  You are using Hidden Markov Model Version: 1
22:13:52   LOG/ DATA-PARSE:  All ERROR tags are parsed successfully.
22:13:52   LOG/ LANGUAGE-MODEL:  Frequency Dictionary for 1-Gram is built successfully.
22:13:52   LOG/ LANGUAGE-MODEL:  Frequency Dictionary for 2-Gram is built successfully.
22:13:52   LOG/ MISSPELLING:  Misspelling (INSERTION-DELETION-SUBSTITUTION) calculation is started.
22:16:16   LOG/ MISSPELLING:  Misspelling (INSERTION-DELETION-SUBSTITUTION) calculation is completed for every ERROR word.
22:16:16   LOG/ HIDDEN-MARKOV-MODEL:  Hidden Markov Model with Viterbi Algorithm has started.
22:16:48   LOG/ HIDDEN-MARKOV-MODEL:  Hidden Markov Model with Viterbi Algorithm finished for each sentences.
22:16:48   LOG/ MAIN-PROCESS:  Spelling Correction program is finished in 176.84334993362427 seconds.
```
