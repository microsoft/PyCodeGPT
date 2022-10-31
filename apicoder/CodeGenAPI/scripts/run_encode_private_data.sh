# !/bash/bin

DOMAIN="PrivateLibrary"
# [train, valid]
SPLIT="train"
CONTAIN_BUILD_IN="False"
# [True, False]
IS_DEBUG="False"
# v1: normal
# v2: # [start] ...
# v3：# Please use these APIs ...
STYLE="v2" # v2, v3, ... vn
PERTURBATION_PROBABILITY=0.05 # [0.1 ~ 1.0]

DATA_DIR="data/Cleaned-Private-Code-Files"
PRIVATE_DATA_DIR="data/API-Doc"

if [ $IS_DEBUG == "True" ]; then
    PRIVATE_LIBS="pandas,numpy,django"
    BUILD_IN_LIBS="datetime"
else
    PRIVATE_LIBS="pandas,numpy,sklearn,torch,tensorflow,django,selenium,matplotlib,flask,scipy,seaborn,nltk,beautifulsoup,pygame,PIL,jieba,gensim,spacy,transformers,fairseq,sqlalchemy,scrapy,allennlp,datasets,tokenizers,mxnet,imageio,pytest,metpy,ansible,requests"
    BUILD_IN_LIBS="datetime,zlib,random,math,sys,glob,os,urllib,time,re,json,unittest,collections,subprocess,copy,functools,itertools,six,threading,tempfile,io,pickle,pathlib,socket,struct,hashlib,traceback,csv,uuid,pprint"
fi

MODEL_DIR="Your/models/codegen/checkpoints/codegen-350M-mono"
OUTPUT_DIR="data/EncodedCorpus4CodeGenAPI"

if [ $IS_DEBUG == "True" ]; then
    N_CPUS="1"
else
    N_CPUS="8"
fi

if [ ! -z "$1" ]; then
    N_CPUS="$1"
fi

Args="-i $DATA_DIR --private_data_path ${PRIVATE_DATA_DIR} -o $OUTPUT_DIR -model $MODEL_DIR -t $N_CPUS -d $DOMAIN --private_libs ${PRIVATE_LIBS} --build_in_libs ${BUILD_IN_LIBS} -isdebug $IS_DEBUG --contain_build_in $CONTAIN_BUILD_IN -pp $PERTURBATION_PROBABILITY --style $STYLE"
echo "Run encode_private for ${SPLIT} data: $Args"

python encode_private_data.py $Args -split ${SPLIT}
echo "Done!"
