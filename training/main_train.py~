"""
Author:         Shraey Bhatia
Date:           October 2016
File:           main_train.py

This is the file which has all the parameters needed to train our model from scratch. A lot of parameters
will need files to be downloaded from URLS provided in readme.

A XML dump is needed to start training from scratch.
Put the path to Wikiextractor(wiki_extractor_path), Stanford tokeiniser(loc_parser) in this file if different
from one menrioned in readme.md

To run thwe whole model with all paramters . python main_train.py -e -td -dv-ng -wv

-e  The extract parameter which will call Wiki-Extractor to process our XML dump file.
- td This tokenizes the proceesed dump documents. It uses stanford tokeniser. This processed documents will be
used to train Doc2vec model. 
-dv This parameter trains the doc2vec model. 
-ng It gets directory of sub bdirectories which contains tokenised documents and following similar directory
    structure  gets documnts which contains wikipedia titles.
-wv Trains word2vec model.
"""

import os
import argparse
parser = argparse.ArgumentParser()


# Parameters for extract.py
wiki_extractor_path = "support_packages/WikiExtractor.py" # Give the Path to WikiExtractor.py file after you download it. URL given in readme.
input_dump = "dump/convertedfile4" # The path to your Wikipedia XML dump.
size ="500M"             # Size of each individual file extracted (you can vary file sizes)
template ="no-templates" # Does not allow WikiExtractor to use any pre installed templates (avoid changing it till you are sure)
output_processed_directory = "processed_documents/docs"   # output directory whre you want documents extracted from dump (path for the directory)

parser.add_argument("-e", "--extract", help="extract wikidump into documents using WikiExtractor",
                    action="store_true")


# Parameters for tokenisation.py
loc_parser = "support_packages/stanford-parser-full-2014-08-27" # Directory for stanford parser.  Download it.
input_directory_to_tokenize = "processed_documents/docs" # Input directory which has WikiExtractor extracted documents i.e output_processed_directory from extract. The main parent directory 
output_directory = "processed_documents/docs_tokenised"  # Path to output directory. It will be created as part of script. 

parser.add_argument("-td", "--tokenize", help="tokenize the documents by stanford parser",
                    action="store_true")


#Parameters for doc2vectrain.py
epochs_doc2vec = 20 # number of training epochs for Doc2Vec model. If you want model to be trained quicker reduce the epochs.
documents_tokenised= "processed_documents/docs_tokenised" # This is the directory where you have all tokenized wikipedia files i.e. output of tokenisation.
output_dir_doc2vec = "trained_models/doc2vec" # Name of directory in which you want to save the trained doc2vec. This directory will be created as part of script.

parser.add_argument("-dv", "--doc2vectrain",help= "train the Doc2Vec model", action = "store_true")


#parameters for create_ngrams.py
# The file which has list of valid n-grams. URL given to download it or can generate it using word2vec_phrase.py
word2vec_phrase_file = "additional_files/word2vec_phrases_list_tokenized.txt" 
input_dir_ngrams = "processed_documents/docs_tokenised" # The directory which has tokensied documents. Already created above in tokenisation
output_dir_ngrams = "processed_documents/docs_ngram" # This is where output will be stored. Will be created in the script.

parser.add_argument("-ng","--ngrams", help ="Run the ngrams file and generate token file with ngrams  (upto 4 gram phrases in it) ", action ="store_true")


#parameters for word2vectrin.py
epochs_word2vec =100 #Ideal number of epochs for word2vecmodel. If you want the model to be trained quicker reduce the epochs.
input_dir_word2vec ="processed_documents/docs_ngram" # This is the directory where you have all tokenized wikipedia files. The script takes all files in the directory for training.
output_dir_word2vec = "trained_models/word2vec" #name of directory in which you want to save trained word2vec. This directory will be created as part of script.

parser.add_argument("-wv", "--word2vectrain", help = "train the word2vec model", action ="store_true")


args = parser.parse_args()

if args.extract:  
    query1 = "python extract.py "+wiki_extractor_path+" "+input_dump +" "+size+" "+template+" " +output_processed_directory
    print query1
    os.system(query1)

if args.tokenize:
    query2 = "python tokenisation.py "+loc_parser +" "+input_directory_to_tokenize+" "+output_directory
    print query2
    os.system(query2)

if args.doc2vectrain:
    query3 = "python doc2vectrain.py "+str(epochs_doc2vec) +" "+documents_tokenised+" "+output_dir_doc2vec
    print query3
    os.system(query3)

if args.ngrams:
   query4 = "python create_ngrams.py "+word2vec_phrase_file+" "+input_dir_ngrams +" "+output_dir_ngrams
   print query4
   os.system(query4)

if args.word2vectrain:
    query5 = "python word2vectrain.py "+str(epochs_word2vec) +" "+input_dir_word2vec+" "+output_dir_word2vec
    print query5
    os.system(query5)
