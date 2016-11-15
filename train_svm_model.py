"""
Author:         Shraey Bhatia
Date:           October 2016
File:           train_svm_model.py

This file is to generate the trained svm model. You can specify your own datset. By default 
will take our dataset. Note a trained svm model for our datset is already in place in model_run/support_files.
You will need binary svm_learn from SVM rank. URL probvided in readme. Update the path here if different.
"""

from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import os
import re
from scipy.spatial.distance import cosine

#Global parameters for the model.

path_svm_learn = "model_run/support_files/svm_rank_learn" # path to learn SVM ranker binary file. You will need to download it. 
output_svm_model = "svm_model"    # name and location to put trained SVM model.
path_pagerank = "model_run/support_files/pagerank-titles-sorted.txt" # Path to pagerank file.
topic_data = "dataset/topics.csv" # The toopic datset. it conatains all topic terms.
topic_labels = "dataset/annotated_dataset.csv" # This is the label datset. It has 19 candidate labels for each topic.
svm_hyperparameter = 0.1  # The SVM hyperparameter.

# reading in topic terms.
topics = pd.read_csv(topic_data)
try:
    new_frame= topics.drop('domain',1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')

# Reading in topic labels.
topic_labels = pd.read_csv(topic_labels, sep ="\t")
topic_labels_without_topic_id= list(topic_labels)
topic_labels_without_topic_id.remove('topic_id')
topic_labels['total'] = topic_labels[topic_labels_without_topic_id].sum(axis=1)
num_raters =topic_labels.count(axis=1) - 3
topic_labels['avg'] = topic_labels['total']/ num_raters

topic_groups = topic_labels.groupby('topic_id')

labels_list =[]
for group in topic_groups:
    temp2 =[]
    temp =list(group[1].label)
    for elem in temp:
        elem = elem.replace(" ","_")
        temp2.append(elem)
    labels_list.append(temp2)

# Reading in pageranks and converting  it into a dictionary.
f2 = open(path_pagerank,'r')
p_rank_dict ={}
for line in f2:
    word = line.split()
    p_rank_dict[word[1].lower()] = word[0]
print "page Rank model loaded"

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
    topic_ls = get_topic_lt(topic_list[num])
    val_dict = {}
    val_list =[]
    final_list=[]
    for item in lab_list:
        trigrams = [item[i:i+3] for i in range(0, len(item) - 2)] #Letter trigram for candidate label.
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
        val = 1 - cosine(np.array(listtopic),np.array(listlabel)) # Cosine Similarity
        val_list.append((item,val))
    rank_val = [i[1] for i in val_list]
    arr = np.array(rank_val)
    order = arr.argsort()
    ranks = order.argsort()
    for i,elem in enumerate(val_list):
        final_list.append((elem[0],ranks[i],int(num)))

    return final_list

# Generates letter trigram feature
temp_lt =[]
for j in range(0,len(topic_list)):
    temp_lt.append(get_lt_ranks(labels_list[j],j))
letter_trigram_feature = [item for sublist in temp_lt for item in sublist]
print "Letter trigram feature generated"

# Changes the format of letter trigram into a dict of dict.
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
Pagerank, letter trigram, Topic overlap and Number of words in a label. Additionally DataFrame will also be given
the label name, topic_id and an avg_val which is average annotator value. This annotator avlue is calculated from 
the candidate label datset and is used to train the SVM model.
"""

def prepare_features(letter_tg_dict,page_rank_dict,cols,feature_names):
    frame =pd.DataFrame()
  
    for x in range(0,len(letter_tg_dict)):
        a = letter_tg_dict[x]
        temp_frame=pd.DataFrame()
        for k in a:
            new_list =[] # The list created to get values for dataframe.
            new_list.append(k) # Candidate label name 
            new_list.append(x) # Topic_id
            temp_val = a[k] # letter trigram feature

            new_list.append(temp_val)
            try:
                pagerank = page_rank_dict[k] #Page Rank Feature
                pagerank = float(pagerank)
            except:
                pagerank = np.nan

            new_list.append(pagerank)
            word_labels = k.split("_")
            com_word_length = len(set(word_labels).intersection(set(topic_list[x]))) # Topic overlap feature
            lab_length = len(word_labels) #Num of words in candidate label feature
            new_list.append(lab_length)
            new_list.append(com_word_length)
            t_label = k.replace("_"," ")
            val = topic_labels[(topic_labels['topic_id'] == x) & (topic_labels['label'] == t_label)]['avg'].values[0] #The annotator value.
            new_list.append(val)
            temp = pd.Series(new_list,index =cols)
            temp_frame = temp_frame.append(temp,ignore_index =True)
            temp_frame = temp_frame.fillna(0)
        for item in feature_names:
            temp_frame[item] = (temp_frame[item] - temp_frame[item].mean())/\
            (temp_frame[item].max() - temp_frame[item].min())  #Feature normalization per topic.
        frame = frame.append(temp_frame,ignore_index =True)
    return frame


cols = ['label','topic_id','letter_trigram','prank','lab_length','common_words','avg_val'] # Name of columns in DataFrame
features =['letter_trigram','prank','lab_length','common_words'] # Feature names

# This function converts the dataset into a format which is taken by SVM ranker.
def convert_dataset(train,feature_names):
    train_list=[]

    for i in range(len(train)):
        
        mystring = str(train[i:i+1]["avg_val"].values[0]) + " "+"qid:"+str(int(train[i:i+1]["topic_id"].values[0]))
        for j,item in enumerate(feature_names):
            mystring = mystring + " "+str(j+1)+":" +str(train[i:i+1][item].values[0])
        mystring = mystring +" # "+train[i:i+1]['label'].values[0]  
        train_list.append(mystring)
    return train_list


feature_dataset =prepare_features(lt_dict,p_rank_dict,cols,features)
print "\n"
print "All features generated"

train_list = convert_dataset(feature_dataset,features)
print "\n"
print "Preparing for generating SVM rank model"

# This method generates the trained SVM file using SVM ranker learn,
def generate_svmrank(train_set):
    h = open("train_temp.dat","w")
    for item in train_set:
        h.write("%s\n" % item)
    h.close()
    query = path_svm_learn +" -c "+str(svm_hyperparameter) +" train_temp.dat "+output_svm_model
    os.system(query)
    query2 = "rm train_temp.dat" # Delete temporary file created.
    os.system(query2)

generate_svmrank(train_list)
