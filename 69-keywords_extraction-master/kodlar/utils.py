import re
import pandas as pd
import numpy as np
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from nltk.tokenize import sent_tokenize
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from tqdm import tqdm
pd.set_option('display.max_colwidth', 200)

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""
  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence
  prefix = ""
  modifier = ""

  #############################################################
  
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]

def get_relation(sent):
  doc = nlp(sent)
  # Matcher class object 
  matcher = Matcher(nlp.vocab)
  #define the pattern 
  pattern = [
            {'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            #{'POS':'ADJ','OP':"?"},
            #{'POS':'NOUN','OP':"?"},
            {'POS':'PROPN','OP':"?"}] 

  matcher.add("matching_1", None, pattern) 
  matches = matcher(doc)
  k = len(matches) - 1
  span = doc[matches[k][1]:matches[k][2]] 
  return(span.text)
  
# Before we build the graph, we need some helper functions.
import re
def is_word(token):
    """
    A token is a "word" if it begins with a letter.
    This is for filtering out punctuations and numbers.
    """
    return re.match(r'^[A-Za-z].+', token)
# We only take nouns and adjectives. See the paper for why this is recommended.
def is_good_token(tagged_token):
    ACCEPTED_TAGS = {'DET','CONJ', 'NUM', 'ADV','PRT', 'C','X'}  
    """
    A tagged token is good if it starts with a letter and the POS tag is
    one of ACCEPTED_TAGS.
    """    
    return is_word(tagged_token[0]) and tagged_token[1] not in ACCEPTED_TAGS
def normalized_token(token):
    """
    Use stemmer to normalize the token.
    """
    return stemmer.stem(token.lower())


def find_punct(text):
    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)
    string="".join(line)
    return list(string)

def cleanup(s,stopset):
    tokens = nltk.word_tokenize(s)
    tagged_tokens = nltk.pos_tag(tokens)
    token = []
    for i in tagged_tokens:
        if is_good_token(i):
            token.append((i[0]))
    cleanup = " ".join(filter(lambda word: word not in stopset, token))
    return cleanup


def get_score_source(source_list,score_table):
    source_result = []
    source_score = []
    for i in source_list:
        i = i.split()
        source_word = pd.DataFrame(i, columns = ['word'])
        source_word = pd.merge(source_word,score_table, on = 'word',how='left')
        source_word = source_word.fillna(0)
        word = ' '.join(i for i in list(source_word['word']))
        score = np.mean(source_word['score'])
        source_result.append(word)
        source_score.append(score)
    return source_result, source_score
def get_score_target(target_list,score_table):
    target_result = []
    target_score = []
    for i in target_list:
        i = i.split()
        target_word = pd.DataFrame(i, columns = ['word'])
        target_word = pd.merge(target_word,score_table, on = 'word',how='left')
        target_word = target_word.fillna(0)
        word = ' '.join(i for i in list(target_word['word']))
        score = np.mean(target_word['score'])
        target_result.append(word)
        target_score.append(score)
    return target_result, target_score