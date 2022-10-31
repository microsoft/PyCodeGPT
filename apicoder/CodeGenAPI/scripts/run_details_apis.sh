#!/bin/bash

# for example: "pandas,numpy,sklearn,tensorflow,keras"
# PS: comma after the comma can not have space

# Third party libraries
LIBRARIES="${LIBRARIES},pandas,numpy,sklearn,torch,tensorflow,django,selenium,matplotlib,flask,scipy,seaborn,nltk,beautifulsoup,pygame,PIL,jieba,gensim,spacy,transformers,fairseq,sqlalchemy,scrapy,allennlp,datasets,tokenizers" 
LIBRARIES="${LIBRARIES},mxnet,imageio,pytest,metpy,ansible,requests"
# Built-in libraries
# LIBRARIES="${LIBRARIES},datetime,zlib,random,math,sys,glob,os,urllib,time,re,json,unittest,collections,subprocess,copy,functools,itertools,six,threading"
# LIBRARIES="${LIBRARIES},tempfile,io,pickle,pathlib,socket,struct,hashlib,traceback,csv,uuid,pprint"

OUTPUT_DIR="data/API-Doc"
PROCESS_NUM=16

Run_Args="-o ${OUTPUT_DIR}"
Run_Args="${Run_Args} -l ${LIBRARIES}"
Run_Args="${Run_Args} -pn ${PROCESS_NUM}"

echo "Run_Args: ${Run_Args}"

python run_extract_details_from_apis.py ${Run_Args}
