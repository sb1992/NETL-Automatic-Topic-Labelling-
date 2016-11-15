"""
Author:         Shraey Bhatia
Date:           October 2016
File:           get_indices.py


This file takes in the output of pruned_documents.py and word2vec_phrases.py and give back the respective indices from doc2vec model and word2vec model 
respectively. You can also download these output files from URLs in readme.(Word2vec phrase List and Filtered/short document titles). Though all these files have already been given to run the models but script is given if you want to create your own. These indices fileswill be used in cand-generation.py to generate label candidates
"""

import os
import gensim
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import math
import re
import pickle

#Global Parameters
doc2vec_model = "model_run/pre_trained_models/doc2vec/docvecmodel.d2v"   # Trained Doc2vec Model
word2vec_model = "model_run/pre_trained_models/word2vec/word2vec" # Trained word2vec model
short_label_documents = "short_label_documents" # The file created by pruned_documents.py. FIltering short or long title documents.
short_label_word2vec_tokenised = "training/additional_files/word2vec_phrases_list_tokenized.txt" #The file created by word2vec_phrases.py Removing brackets from filtered wiki titles.
doc2vec_indices_output = "doc2vec_indices"  # The output file which map pruned doc2vec labels to indcies from doc2vec model.
word2vec_indices_output ="word2vec_indices" # the output file that maps short_label_word2vec_tokenised to indices from wrod2vec model.

# Removing any junk labels and also if a label pops up with the term disambiguation.
def get_word(word):
    inst = re.search(r"_\(([A-Za-z0-9_]+)\)", word)

    if inst == None:
        length = len(word.split("_"))
        if length < 5:
            return True, word
    else:
        if inst.group(1) != "disambiguation":
            word2 = re.sub(r'_\(.+\)','',word)
            if len(word2.split(" ")) <5:
                return True, word

    return False,word

# Load the trained doc2vec and word2vec models.
model1 =Doc2Vec.load(doc2vec_model)
model2 = Word2Vec.load(word2vec_model)
print "Models loaded"

# Loading the pruned tiles and making a set of it
with open(short_label_documents,"r") as k:
    doc_labels = pickle.load(k)
doc_labels = set(doc_labels)
print "Pruned document titles loaded"

# laoding thw phrasses used in training word2vec model. And then replacing space with underscore.
h = open(short_label_word2vec_tokenised,'r')
list_labels=[]
for line in h:
    line = line.strip()
    list_labels.append(line)
list_labels= set(list_labels)

word2vec_labels=[]
for words in list_labels:
    new = words.split(" ")
    temp ='_'.join(new)
    word2vec_labels.append(temp)
word2vec_labels = set(word2vec_labels)
print "Word2vec model phrases loaded"

doc_indices =[]
word_indices =[]

# finds the coresponding index of the title from doc2vec model
for elem in doc_labels:
    status,item = get_word(elem)
    if status:
        try:
            val = model1.docvecs.doctags[elem].offset
            doc_indices.append(val)
        except:
            pass

# Finds the corseponding index from word2vec model
for elem in word2vec_labels:
    try:
        val = model2.vocab[elem].index
        word_indices.append(val)
    except:
        pass

# creating output indices file
with open(doc2vec_indices_output,'wb') as m:
    pickle.dump(doc_indices,m)
with open(word2vec_indices_output,'wb') as n:
    pickle.dump(word_indices,n)

