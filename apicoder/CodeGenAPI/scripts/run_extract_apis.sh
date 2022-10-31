#!/bin/bash

export DJANGO_SETTINGS_MODULE=bay.settings # 

# for example: "pandas,numpy,sklearn,tensorflow,keras"
# PS: behind the comma can not have space
# Third party libraries
LIBRARIES="${LIBRARIES},pandas,numpy,sklearn,torch,tensorflow,django,selenium,matplotlib,flask,scipy,seaborn,nltk,beautifulsoup,pygame,PIL,jieba,gensim,spacy,transformers,fairseq,sqlalchemy,scrapy,allennlp,datasets,tokenizers,torchdata" 
LIBRARIES="${LIBRARIES},mxnet,imageio,pytest,metpy,ansible,requests"
# Built-in libraries
# LIBRARIES="${LIBRARIES},datetime,zlib,random,math,sys,glob,os,urllib,time,re,json,unittest,collections,subprocess,copy,functools,itertools,six,threading"
# LIBRARIES="${LIBRARIES},tempfile,io,pickle,pathlib,socket,struct,hashlib,traceback,csv,uuid,pprint"

ID=$(date +"%m%d")
OUTPUT_DIR="data/API-Doc"
PROCESS_NUM=16
OVER_WRITE="True" # [True, False]
GET_SIG="True" # [True, False]

Run_Args="-o ${OUTPUT_DIR}"
Run_Args="${Run_Args} -ls ${LIBRARIES}"
Run_Args="${Run_Args} -pn ${PROCESS_NUM}"
Run_Args="${Run_Args} -ow ${OVER_WRITE}"
Run_Args="${Run_Args} -gs ${GET_SIG}"

echo "Run_Args: ${Run_Args}"

python -u extract_api.py ${Run_Args}
