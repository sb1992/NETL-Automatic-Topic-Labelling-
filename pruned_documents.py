"""
Author:         Shraey Bhatia
Date:           October 2016
File:           pruned_documents.py

This file give us a list of Wikipedia titles(hence the documents associated with those titles) which we want to consider from our
Doc2vec trained model and for further computations in running the model. This file is also used in generating output of doc2vec_indices.py used in 
model_run/candidate-generation.py. If nneded this file is already given in URL in readme Filtered/short Document titles. 

"""

import re
import os
import pickle
import codecs
import unicodedata
import multiprocessing as mp
from multiprocessing import Pool

#Gobals
title_length = 5 # Length of wikipedia title you want to filter. All titles greater than or equal to this value will be thrown away
doc_length = 40 # length of documents. All documents having number of words less than this value will not be considered.
tokenised_wiki_directory = 'training/processed_documents/docs_tokenised' # the directory in which you tokenised all files extracted from Wiki-Extractor using stanford tokenizer
output_filename = 'short_label_documents' # The name of output file you want the list of valid wikipedia titles to be saved into. Will be a pickle file


def get_labels(filename):
    list_labels =[]
    f= codecs.open(filename, "r", "utf-8")
    for line in f:
        if "<doc" in line:
            
            found =""
            m= re.search('title="(.*)">',line)   # Uses regular expression to get wiki title. The title is in this format if you use Wiki-Extractor.
            try:
                found = m.group(1)
                found = unicodedata.normalize("NFKD", found) # next few steps are encodeing and ecoding betwen utf and unicode
                found = found.replace(" ","_")
                found = found.encode('utf-8')
            except:
                found =""
            values=[]
        else:
            if found != "":                    
                if "</doc" not in line:
                    for word in line.split(" "):
                        values.append(word.strip())

                if "</doc" in line:                     # checks if we reach end of that particular document and if condition ois satisfied title is added into list. 
                    temp_list= found.split("_")
                    if (len(values) > doc_length) and (len(temp_list) < title_length):
   		        list_labels.append(found)
   
    return list_labels

# Walking through directory and getting the filenames from the tokenised directory.
filenames=[]
for path,subdirs,files in os.walk(tokenised_wiki_directory):
    for name in files:
        temp = os.path.join(path, name)
        filenames.append(temp)

print "Got all files"
# Multiprocess files
cores = mp.cpu_count()
pool = Pool(processes = cores)
y_parallel = pool.map(get_labels, filenames)

# converting a list of list into list
all_docs = [item for sublist in y_parallel for item in sublist]

#Writing into pickle file
print "Writng labels to picke file"
with open(output_filename,'w') as k:
    pickle.dump(all_docs,k)
