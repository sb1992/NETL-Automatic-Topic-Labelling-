"""
Author:         Shraey Bhatia
Date:           October 2016
File:           get_labels.py

It is the script to generate candidate labels, unsupervised best labels and labels from SVM ranker supervised model. 
Update the parameters in this script. 
Also after you download the files mentioned in readme and if you keep them in different path change it over here.

Parameters:
-cg To generate candiidate labels.
-us To get unsupervised labels.
-s To get supervised labels.
Ideally should first use -cg to get candiate label file before going for unsupervised or supervised model. But can be used directly if
you already have your candidate label file for the topics. 
Eamample for topics given in toy_data/toytopics.csv
"""

import os
import argparse
parser = argparse.ArgumentParser()

#Common Parameters
data ="toy_data/toytopics.csv" # The file in csv format which contains the topic terms that needs a label. 

#Parameters for candidate Generation of Labels
doc2vecmodel = "pre_trained_models/doc2vec/docvecmodel.d2v" # Path for Doc2vec Model.
word2vecmodel = "pre_trained_models/word2vec/word2vec" # Path for Word2vec Model.
num_candidates =19 # Number of candidates labels that need to generated for a topic.
output_filename = "output_candidates" # Name of the output file that will store the candidate labels.
doc2vec_indices_file = "support_files/doc2vec_indices" # The filtered doc2vec indices file. 
word2vec_indices_file = "support_files/word2vec_indices" # The filtered word2vec indices file
parser.add_argument("-cg", "--candidates", help ="get candidate labels", action = "store_true")

#Unsupevised model parameters
num_unsup_labels =3 # Number of unsupervised labels needed (In general should be less than candidate labels, till you have your own file and then depends on number of labels)
cand_gen_output = "output_candidates" # The file which contains candiate generation output.Also used to get supervised output
out_unsup = "output_unsupervised" # The Output File name for unsupervised labels
parser.add_argument("-us", "--unsupervised", help="get unsupervised labels", action="store_true")

#supervised parameters
num_sup_labels = 3 # Number of supervised labels needed. Should be less than the candidate labels.
pagerank_model = "support_files/pagerank-titles-sorted.txt" # This is precomputed pagerank model needed to genrate pagerank features.
svm_classify = "support_files/svm_rank_classify" # SVM rank classify. After you download SVM Ranker classify gibve the path of svm_rank_classify here
pretrained_svm_model = "support_files/svm_model" # This is trained supervised model on the whole our dataset. Run train train_svm_model.py if you want a new model on different dataset. 
out_sup ="output_supervised" # The output file for supervised labels.
parser.add_argument("-s", "--supervised", help ="get supervised labels", action ="store_true")

args = parser.parse_args()
if args.candidates:  # It calls unsupervised_labels python file to get labels in unsupervised way
    query1 = "python cand_generation.py "+str(num_candidates)+" "+doc2vecmodel+" "+word2vecmodel+" "+data+" "+output_filename +" "+doc2vec_indices_file+" "+word2vec_indices_file
    print "Extracting candidate labels"
    os.system(query1)

if args.unsupervised:  # It calls unsupervised_labels python file to get labels in unsupervised way
    query2 = "python unsupervised_labels.py "+str(num_unsup_labels)+" "+data+" "+cand_gen_output +" "+out_unsup
    print "Executing Unsupervised model"
    os.system(query2)

if args.supervised:  # It calls supervised_labels python file to get labels in supervised way.
    query3 = "python supervised_labels.py " +str(num_sup_labels)+" "+pagerank_model+" "+data+" "+cand_gen_output+" "+svm_classify+" "+pretrained_svm_model+" "+out_sup
    print "Executing Supervised Model"
    os.system(query3)

