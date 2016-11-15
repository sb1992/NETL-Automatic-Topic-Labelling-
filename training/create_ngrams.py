"""
Author:         Shraey Bhatia
Date:           October 2016
File:           create_ngrams.py

Takes in tokenised documents and replace it with phrases.Replacement occurs if ngrams ( till n=4) 
are valid wikipedia titles. Theses consecutive words are joined together with underscore
in betwwen words to tell that it acts as a phrase. The output directory will have same structure as input.
It takes in a word2vec prhase list file which can be downloaded and placed in additional_files or generated from word2vec_phrases.py
parameters taken from main_train.py           
"""

import re
import sys
import multiprocessing as mp
import pickle
import os
from multiprocessing import Pool
import nltk
from nltk.util import ngrams
import argparse

# The text file of all wiki titles which are less than or equal to 4 words.
#h = open('short_labels_list_tokenized.txt','r')

parser = argparse.ArgumentParser()
parser.add_argument("word2vec_phrases")
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()

# Checks if the output directory specified already exists. If it remove it.
if os.path.isdir(args.output_dir):
    del_query = "rm -r "+args.output_dir
    os.system(del_query)


# Creating output directory.
query = "mkdir "+args.output_dir
os.system(query)

# The text file of all wiki titles which are less than or equal to 4 words. 

h = open(args.word2vec_phrases)
list_labels=[]
for line in h:
    line = line.strip()
    list_labels.append(line)
list_labels= set(list_labels)
print "file loaded"


def get_phrases(filenames):
    input,output = filenames
    cnt =0
    print "output file name"
    print output
    f = open(input, 'r')
    doc_number =0
    lines =[]
    for line in f:
        if "<doc" in line:
            doc_number =doc_number +1        # Just to keep a track that documents are being processed and their count.
            if (doc_number % 5000 ==0):      # Print number every 500 documents. Since files are multiprocessed count numbers can be repititive.
                print "Documents processed "+str(doc_number)
        else:
            words = line.split(" ")
            bigram = ngrams(words,2)
            trigram = ngrams(words,3)
            fourgram = ngrams(words,4)
            
            for item in fourgram:
                temp = ' '.join(item)
                if temp in list_labels:
                    temp1 = '_'.join(item)
                    line  = line.replace(temp,temp1)
                    

            for item in trigram:
                temp = ' '.join(item)
                if temp in list_labels:
                    temp1 = '_'.join(item)
                    line  = line.replace(temp,temp1)
            for item in bigram:
                temp = ' '.join(item)
                if temp in list_labels:
                    temp1 = '_'.join(item)
                    line  = line.replace(temp,temp1)            

        lines.append(line)

    
    with open(output,'w') as g:
        g.write('\n'.join(lines))
  
list_files = os.listdir(args.input_dir)
inp_filenames =[]
out_filenames =[]

for item in list_files:

    inp_subdir =args.input_dir +"/"+ item # Getting the full path for subdirectories.
    subfiles = os.listdir(inp_subdir)     # listing the file in subdirectory
    out_subdir = args.output_dir +"/"+ item
    query = "mkdir " +out_subdir   # Making new sub directories in output location, so that the directory structure is same as input directoryy
    os.system(query)

    for elem in subfiles:
        input_file = inp_subdir + "/"+elem  
        output_file = out_subdir + "/"+elem+"_ngram"
	inp_filenames.append(input_file) # Getting path of all files that needs to be converted in ngrams.
        out_filenames.append(output_file)# The output names with for files with path.

print "Got all files"
print "Converting to ngram phrases"

cores = mp.cpu_count()
pool = Pool(processes = cores)
y_parallel = pool.map(get_phrases,zip(inp_filenames,out_filenames))

