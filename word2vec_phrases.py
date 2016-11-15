"""
Author:         Shraey Bhatia
Date:           October 2016
File:           word2vec_phrases.py

This file give us a list of Wikipedia titles which has to be used as ngrams in running word2vec model. This removes brackets from title
and all filter with ttile length and document size. This is the file to be used in ngramsgen.py. Stanford Parser is used to tokenise the files
as replacement has to be occured on tokenised files with the same tokeniser. 
The output of this file is also used in generating output of word2vec_indices.py which is used in model_run/candidate-generation.py.
One such file; URL is in readme Word2vec Phrase list. You can update parameters in this file.
"""

import re
import os
import pickle
import codecs
import unicodedata
import multiprocessing as mp
from multiprocessing import Pool

#Gobals
title_length = 5 # Length of wikipedia title you want to filter. All titles greter than or equal to this value will be thrown away
doc_length = 40 # length of documents. All documents having number of words less than this value will not be considered.
tokenised_wiki_directory = 'training/processed_documents/docs_tokenised' # the directory in which you tokenised all files extracted from Wiki-Extractor using stanford tokenizer
output_filename = 'training/additional_files/word2vec_phrases_list_tokenized2.txt' # The name of output file you want the list of valid wikipedia titles to be saved into. Will be a pickle file
loc_parser = "training/support_packages/stanford-parser-full-2014-08-27" # Directory of stanford Parser.

classpath = loc_parser +"/stanford-parser.jar"  # Full classpath for jar file

# Method removes parenthesis brackets from labels
def get_word(word):
    inst = re.search(r" \((.+)\)", word)
    if inst == None:
        return word
    else:
        word = re.sub(r' \(.+\)','',word)
        return word


def get_labels(filename):
    list_labels =[]
    f= codecs.open(filename, "r", "utf-8")
    for line in f:
        if "<doc" in line:
            
            found =""
            m= re.search('title="(.*)">',line)   # Uses regular expression to get wiki title. The title is in this format if you use Wiki-Extractor.
            try:
                found = m.group(1)
                found = unicodedata.normalize("NFKD", found) # next few steps are encodeing and decoding betwen utf and unicode
      
            except:
                found =""
            values=[]
        else:
            if found != "":                    
                if "</doc" not in line:
                    for word in line.split(" "):
                        values.append(word.strip())

                if "</doc" in line:                     # checks if we reach end of that particular document and if condition ois satisfied title is added into list. 
                    temp_list= found.split(" ")
                    if (len(values) > doc_length) and (len(temp_list) < title_length):
                        found = get_word(found) #Removing brackets if present
                        found = found.encode('utf-8')
              
   		        list_labels.append(found)
    return list_labels

# Walking through directory and getting the filenames from the tokenised directory.
filenames=[]
for path,subdirs,files in os.walk(tokenised_wiki_directory):
    for name in files:
        temp = os.path.join(path, name)
        filenames.append(temp)
filenames_temp = filenames[:4]
print "Got all files"

# Multiprocess files
cores = mp.cpu_count()
pool = Pool(processes = cores)
y_parallel = pool.map(get_labels, filenames_temp)

# converting a list of list into list
all_docs = [item for sublist in y_parallel for item in sublist]

#Writing list of titles into temporary file which will be okenised
print "Generating a temporary file"

g =open("temp.txt",'w')
set_docs= set(all_docs)
for elem in set_docs:
    g.write(elem +"\n")
g.close()

print "Tokenising"

# Running query for standford parser
query = "java -cp "+ classpath +" edu.stanford.nlp.process.PTBTokenizer -preserveLines --lowerCase <temp.txt> "+output_filename
os.system(query)

# Deleting temporary file
os.system("rm temp.txt")
