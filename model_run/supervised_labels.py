"""
Author:         Shraey Bhatia
Date:           October 2016
File: 		supervised_labels.py

This python code gives the top supervised labels for that topic. The paramters needed are passed through get_labels.py.
It generates letter_trigram,pagerank, Topic overlap and num of words in features. Then puts it into SVM classify
format and finally uses the already trained supervised model to make ranking predictions and get the best label. 
You will need SVM classify binary from SVM rank. The URL is provided in readme.

"""

import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
import os
import sys
import argparse

# Arguments being passed which were in get_labels.py file.
parser = argparse.ArgumentParser()
parser.add_argument("num_sup_labels") # num of supervised labels needed.
parser.add_argument("pagerank_model") # path to the pagerank file
parser.add_argument("data") # path to the topic data file.
parser.add_argument("output_candidates") # path of generated candidate file.
parser.add_argument("svm_classify") # path to the SVM Ranker classify binary file. Needs to be downloaded from the path provided in Readme.
parser.add_argument("trained_svm_model") # This is the pre existing trained SVM model, trained on our SVM model.
parser.add_argument("output_supervised") # Output file for supervised labels
args = parser.parse_args()

# Load the pagerank File nto a dictionary
f2 = open(args.pagerank_model,'r')
p_rank_dict ={}
for line in f2:
    word = line.split()
    p_rank_dict[word[1].lower()] = word[0]
print "page Rank models loaded"

# Get the candidate labels form candiate label file
label_list =[]
with open(args.output_candidates,'r') as k:
    for line in k:
        labels = line.split()
        label_list.append(labels[1:])

# Just get the number of labels per topic.
test_chunk_size = len(label_list[0])

# Number of Supervised labels needed should not be less than the number of candidate labels.
if test_chunk_size < int(args.num_sup_labels):
    print "\n"
    print "Error"
    print "You cannot extract more labels than present in input file"
    sys.exit() 

# Reading in the topic terms from the topics file.
topics = pd.read_csv(args.data)
try:
    new_frame= topics.drop('domain',1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')
print "Data Gathered for supervised model"

# Method to get letter trigrams for topic terms.
def get_topic_lt(elem):
    tot_list =[]
    for item in elem:
        trigrams = [item[i:i+3] for i in range(0, len(item) - 2)] 
        tot_list = tot_list + trigrams
    x = Counter(tot_list)
    total = sum(x.values(), 0.0)
    for key in x:
        x[key] /= total
    return x

"""
This method will be used to get first feature of letter trigrams for candidate labels and then rank them.
It use cosine similarity to get a score between a letter trigram vector of label candidate and vector of 
topic terms.The ranks are given based on that score.

"""

def get_lt_ranks(lab_list,num):
    topic_ls = get_topic_lt(topic_list[num]) # Will get letter trigram for topic terms.
    val_dict = {}
    val_list =[]
    final_list=[]
    for item in lab_list:
        trigrams = [item[i:i+3] for i in range(0, len(item) - 2)] # get the trigrams for label candidate.
        label_cnt = Counter(trigrams)
        total = sum(label_cnt.values(), 0.0)
        for key in label_cnt:
            label_cnt[key] /= total
        tot_keys = list(set(topic_ls.keys() + label_cnt.keys()))
        listtopic = []
        listlabel = []
        for elem in tot_keys:
            if elem in topic_ls:
                listtopic.append(topic_ls[elem])
            else:
                listtopic.append(0.0)
            if elem in label_cnt:
                listlabel.append(label_cnt[elem])
            else:
                listlabel.append(0.0)
        val = 1 - cosine(np.array(listtopic),np.array(listlabel)) # Cosine similarity.
        val_list.append((item,val))
    rank_val = [i[1] for i in val_list]
    arr = np.array(rank_val)
    order = arr.argsort()
    ranks = order.argsort()
    for i,elem in enumerate(val_list):
        final_list.append((elem[0],ranks[i],int(num)))
        
    return final_list

# This calls the above method to get letter trigram feature.
temp_lt =[]
for j in range(0,len(topic_list)):
    temp_lt.append(get_lt_ranks(label_list[j],j))
letter_trigram_feature = [item for sublist in temp_lt for item in sublist] 
print "letter trigram feature"
#print letter_trigram_feature

# Employed to change the format of features.
def change_format(f1):
    lt_dict =defaultdict(dict)
    
    for elem in f1:
        x,y,z = elem
        lt_dict[z][x] = y
    return lt_dict


lt_dict = change_format(letter_trigram_feature)

"""
This method is to prepare all features. It will take in dictionary of letter trigram, pagerank, list of 
all columns for the datframe and name of features. It will generate four features in the dataframe namely
Pagerank, letter trigram, Topic overlap and Number of words in a label. Additionally DatFrame will also be given
the label name, topic_id and an avg_val which is average annotator value. It is just given a value of 3 here 
but can be anything as it does not make a difference in prediction. Only important when we have to train SVM model.
"""

def prepare_features(letter_tg_dict,page_rank_dict,cols,feature_names):
    frame =pd.DataFrame()
    for x in range(0,len(letter_tg_dict)):
        a = letter_tg_dict[x]
        temp_frame=pd.DataFrame()
        for k in a:
            new_list =[]  # The list created to get values for dataframe.
            new_list.append(k)  # The label name
            new_list.append(x) # The topic _id.
            temp_val = a[k]  # letter trigram value
            new_list.append(temp_val)
            try:
                pagerank = page_rank_dict[k]
                pagerank = float(pagerank)
            except:
                pagerank = np.nan
            
            new_list.append(pagerank) # pagerank value
            word_labels = k.split("_") 
            com_word_length = len(set(word_labels).intersection(set(topic_list[x]))) # Extracting topic overlap.
            lab_length = len(word_labels) # number of words in the candidate label.   
            new_list.append(lab_length)
            new_list.append(com_word_length)
            new_list.append(3) #This could be just any value appended for the sake of giving a column for annotator rating neeeded in SVM Ranker classify
            temp = pd.Series(new_list,index =cols)
            temp_frame = temp_frame.append(temp,ignore_index =True)
            temp_frame = temp_frame.fillna(0) # Just filling in case a label does not have a pagerank value. Generally should not happen
        for item in feature_names:
            temp_frame[item] = (temp_frame[item] - temp_frame[item].mean())/\
            (temp_frame[item].max() - temp_frame[item].min())  # feature Normalization per topic.
        frame = frame.append(temp_frame,ignore_index =True)
        frame = frame.fillna(0)
    return frame


cols = ['label','topic_id','letter_trigram','prank','lab_length','common_words','avg_val'] # Name of columns for DataFrame.
features =['letter_trigram','prank','lab_length','common_words'] # Name of features.

feature_dataset =prepare_features(lt_dict,p_rank_dict,cols,features)
print "All features generated"

# This function converts the dataset into a format which is taken by SVM ranker classify binary file.

def convert_dataset(test_file,feature_names):
    test_list=[]

    for i in range(len(test_file)):
        
        mystring = str(test_file[i:i+1]["avg_val"].values[0]) + " "+"qid:"+str(int(test_file[i:i+1]["topic_id"].values[0]))
        for j,item in enumerate(feature_names):
            mystring = mystring + " "+str(j+1)+":" +str(test_file[i:i+1][item].values[0])
        mystring = mystring +" # "+test_file[i:i+1]['label'].values[0]  
        test_list.append(mystring)
    return test_list

test_list = convert_dataset(feature_dataset,features)


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

# It calls SVM classify and gets predictions for each topic. 
def get_predictions(test_set,num):
    h = open("test_temp.dat","w")
    for item in test_set:
        h.write("%s\n" % item)
    h.close()
   
    query2 =args.svm_classify + " test_temp.dat "+args.trained_svm_model+" predictionstemp"
    os.system(query2)
    h =open("predictionstemp")
    pred_list =[]
    for line in h:
        pred_list.append(line.strip())
    h.close()
  
    pred_chunks = chunks(pred_list,num)
    test_chunks = chunks(test_set,num)
    list_max =[]
    for j in range(len(pred_chunks)):
        max_sort = np.array(pred_chunks[j]).argsort()[::-1][:int(args.num_sup_labels)]
        list_max.append(max_sort)
    print "\n"
    print "Printing Labels for supervised model"
    g = open(args.output_supervised,'w')
    for cnt, (x,y) in enumerate(zip(test_chunks,list_max)):
        print "Top "+args.num_sup_labels+" labels for topic "+str(cnt)+" are:"
        g.write( "Top "+args.num_sup_labels+" labels for topic "+str(cnt)+" are:" +"\n")
        for i2 in y:
            m= re.search('# (.*)',x[i2])
            print m.group(1)
            g.write(m.group(1)+"\n")

        print "\n"
        g.write("\n")
    g.close()

    query3 ="rm test_temp.dat predictionstemp"   # deleting the ttest file and prediction file generated as part of code to run svm_classify
    os.system(query3)

get_predictions(test_list,test_chunk_size)
