"""
Author:         Shraey Bhatia
Date:           October 2016
File:           extract.py

This file uses WikiExtractor tool to generate documents from wikipedia xml dump. 
WikExtractor tool can be found at https://github.com/attardi/wikiextractor.
If you use a diffferent path than to one mentioned in readme update it in main_train.py
Arguments for this file are taken from there.
""" 

import os
import argparse
import sys

# The arguments for WikiExtractor. These parameters have been explained in main_train.py

parser = argparse.ArgumentParser()
parser.add_argument("wiki_extractor_path")
parser.add_argument("input_dump") # The Xml dump
parser.add_argument("size")
parser.add_argument("template")
parser.add_argument("output_processed_dir") # the output directory
args = parser.parse_args()

# Checks if the output directory specified already exists. If it does removes it.

if os.path.isdir(args.output_processed_dir):
    del_query = "rm -r "+args.output_processed_dir
    os.system(del_query)

# Creates the output directory.
query1 = "mkdir "+args.output_processed_dir
os.system(query1)
query2 = "python "+args.wiki_extractor_path+" "+args.input_dump +" -o" +args.output_processed_dir +" -b " +args.size +" --"+args.template
os.system(query2)

