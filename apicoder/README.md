# Private Library Oriented Code Generation

This project is the code, data, benchmarks, and models for our paper titled [When Language Model Meets Private Library](https://arxiv.org/pdf/2210.17236.pdf).

---

## Abstract

With the rapid development of pre-training techniques, a number of language models have been pre-trained on large-scale code corpora and perform well in code generation. In this paper, we investigate how to equip the pre-trained language models with the ability of code generation for private libraries. In practice, it is common for programmers to write code using private libraries. However, this is a challenge for language models, since they have never seen private APIs during training. Motivated by the fact that private libraries usually come with elaborate API documentations, we propose a novel framework with two modules: the APIRetriever finds useful APIs, and then the APICoder generates code using these APIs. For APIRetriever, we present a dense retrieval system and also design a friendly interaction to involve uses. For APICoder, we can directly use the off-the-shelf language models like CODEGEN, and we also continually pre-trained the base model on a code corpus containing API information. Both modules are trained with data from public libraries, and can be generalized to private ones. Furthermore, we craft three benchmarks for private libraries, named TorchDataEval, MonkeyEval, and BeatNumEval. Experimental results demonstrate the impressive performance of our framework.

<img src=https://s3.bmp.ovh/imgs/2022/09/27/624d82ddc3045aea.png width=450 />

## Architecture of this repository

As this project consists of a lot of modules, we have drawn the logic diagram between data and code for you to easily understand it.

> You can view this diagram in line with the following project directory at a glance. 

<img src=https://s3.bmp.ovh/imgs/2022/09/27/b81213d3e5d841b2.png width=1000 />

Figure1: Logic flow diagram of our code and data, where the gray background denotes the file (e.g, model, data), the red and blue letters mean the relative path of the file or code.

## Project Directory
```shell
├── APIRetriever
│   ├── LICENSE
│   ├── README.md
│   ├── build
│   ├── data
│   │   ├── inference
│   │   │   ├── beatnum_api.json
│   │   │   ├── beatnum_api.pt
│   │   │   ├── beatnum_comment.json
│   │   │   ├── beatnum_comment.pt
│   │   │   ├── beatnum_id_score.trec
│   │   │   ├── beatnum_id_score.txt
│   │   │   ├── monkey_api.json
│   │   │   ├── monkey_api.pt
│   │   │   ├── monkey_comment.json
│   │   │   ├── monkey_comment.pt
│   │   │   ├── monkey_id_score.trec
│   │   │   ├── monkey_id_score.txt
│   │   │   ├── numpy_api.json
│   │   │   ├── numpy_api.pt
│   │   │   ├── numpy_comment.json
│   │   │   ├── numpy_comment.pt
│   │   │   ├── numpy_id_score.trec
│   │   │   ├── numpy_id_score.txt
│   │   │   ├── pandas_api.json
│   │   │   ├── pandas_api.pt
│   │   │   ├── pandas_comment.json
│   │   │   ├── pandas_comment.pt
│   │   │   ├── pandas_id_score.trec
│   │   │   ├── pandas_id_score.txt
│   │   │   ├── torchdata_api.json
│   │   │   ├── torchdata_api.pt
│   │   │   ├── torchdata_comment.json
│   │   │   ├── torchdata_comment.pt
│   │   │   ├── torchdata_id_score.trec
│   │   │   └── torchdata_id_score.txt
│   │   └── train
│   │       ├── processed-train-data
│   │       └── unprocessed-train-data
│   ├── outputs
│   ├── requirements.txt
│   ├── scripts
│   │   ├── extract_retrieval_api_corpus.py
│   │   ├── run_extract_apiretriever_corpus.sh
│   │   ├── run_prepare_test_private_code.py
│   │   └── run_prepare_train_private_code.py
│   ├── setup.py
│   └── src
│       ├── dense
│       ├── run_encode_2.sh
│       ├── run_search_3.sh
│       ├── run_train_1.sh
│       └── run_trec_format_4.sh
├── CodeGenAPI
│   ├── APICoder
│   │   ├── get_api_info_by_name.py
│   │   └── get_lib_comment_for_eval.py
│   ├── README.md
│   ├── eval_baseline.py
│   ├── eval_private.py
│   ├── nl2code
│   │   ├── __init__.py
│   │   ├── code_dataset.py
│   │   ├── code_dataset_codegen.py
│   │   ├── configuration_codegen.py
│   │   ├── hf_trainer.py
│   │   ├── indexed_dataset.py
│   │   └── modeling_codegen.py
│   ├── requirements.txt
│   ├── run_evaluating_codes.sh
│   ├── run_generating_codes.sh
│   ├── run_private.py
│   ├── run_private.sh
│   └── scripts
│       ├── __init__.py
│       ├── encode_private_data.py
│       ├── extract_api.py
│       ├── file_utils.py
│       ├── get_comments_from_evallibs.py
│       ├── get_libs_info_from_code.py
│       ├── make_human_in_the_loop_test_corpus.py
│       ├── multiprocessing_utils.py
│       ├── pycode_visitor.py
│       ├── requirements.txt
│       ├── run_details_apis.sh
│       ├── run_encode_private_data.sh
│       ├── run_extract_apis.sh
│       └── run_extract_details_from_apis.py
├── README.md
├── data
│   ├── API-Doc
│   │   └── README.md
│   ├── Cleaned-Private-Code-Files
│   │   └── README.md
│   ├── CodeGenAPI
│   │   └── README.md
│   └── EncodedCorpus4CodeGenAPI
│       └── README.md
└── private-eval
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── TorchData_no.API_number_0.CodeGen.hm_False.machine.t0.1.p0.9.l100.n1.samples.jsonl
    │   ├── XXXAPIEval-make sense.ipynb
    │   ├── numpy_keyword_mapping.json
    │   ├── numpy_keywords.jsonl
    │   ├── pandas_keyword_mapping.json
    │   ├── pandas_keywords.jsonl
    │   ├── real_beatnum_eval_v3.jsonl.gz
    │   ├── real_beatnum_eval_v3_api_1.jsonl.gz
    │   ├── real_beatnum_eval_v3_api_2.jsonl.gz
    │   ├── real_beatnum_eval_v3_api_3.jsonl.gz
    │   ├── real_beatnum_eval_v3_api_5.jsonl.gz
    │   ├── real_beatnum_eval_v3_api_n.jsonl.gz
    │   ├── real_beatnum_eval_v3_human_labelled.jsonl.gz
    │   ├── real_monkey_eval_v3.jsonl.gz
    │   ├── real_monkey_eval_v3_api_1.jsonl.gz
    │   ├── real_monkey_eval_v3_api_2.jsonl.gz
    │   ├── real_monkey_eval_v3_api_3.jsonl.gz
    │   ├── real_monkey_eval_v3_api_5.jsonl.gz
    │   ├── real_monkey_eval_v3_api_n.jsonl.gz
    │   ├── real_monkey_eval_v3_human_labelled.jsonl.gz
    │   ├── real_numpy_eval_v3.jsonl.gz
    │   ├── real_numpy_eval_v3_api_1.jsonl.gz
    │   ├── real_numpy_eval_v3_api_2.jsonl.gz
    │   ├── real_numpy_eval_v3_api_3.jsonl.gz
    │   ├── real_numpy_eval_v3_api_5.jsonl.gz
    │   ├── real_numpy_eval_v3_api_n.jsonl.gz
    │   ├── real_pandas_eval_v3.jsonl.gz
    │   ├── real_pandas_eval_v3_api_1.jsonl.gz
    │   ├── real_pandas_eval_v3_api_2.jsonl.gz
    │   ├── real_pandas_eval_v3_api_3.jsonl.gz
    │   ├── real_pandas_eval_v3_api_5.jsonl.gz
    │   ├── real_pandas_eval_v3_api_n.jsonl.gz
    │   ├── real_torchdata_eval_v3.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_1.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_1_make_sense.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_2.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_2_make_sense.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_3.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_3_make_sense.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_5.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_5_make_sense.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_n.jsonl.gz
    │   ├── real_torchdata_eval_v3_api_n_make_sense.jsonl.gz
    │   ├── real_torchdata_eval_v3_human_labelled.jsonl.gz
    │   └── real_torchdata_eval_v3_human_labelled_make_sense.jsonl.gz
    ├── private_eval
    │   ├── __init__.py
    │   ├── data.py
    │   ├── evaluate_functional_correctness.py
    │   ├── evaluation.py
    │   └── execution.py
    ├── requirements.txt
    └── setup.py
```

## Download
Download the API-Doc file below, unzip it and put it under path `data/API-Doc`.

> [Click here to download API-Doc](https://github.com/microsoft/PyCodeGPT/releases/download/Private-Library/API-Doc.zip).
```
cd data/API-Doc
unzip API-Doc.zip
```

You should download the CodeGenAPI file below, unzip it and put it under path `data/CodeGenAPI`.

> [Click here to download CodeGenAPI-350M-mono](https://github.com/microsoft/PyCodeGPT/releases/download/Private-Library/CodeGenAPI-350M-mono.zip)
```
cd data/CodeGenAPI
unzip CodeGenAPI-350M-mono.zip
```

Also, you should download the private API embeddings below, unzip it and put it under path `APIRetriever/data/inference`.

> [Click here to download private API embeddings](https://github.com/microsoft/PyCodeGPT/releases/download/Private-Library/private-api-embeddings.zip)
```
cd APIRetriever/data/inference
unzip private-api-embeddings.zip
```

## Citation
If you find our work useful, please cite the paper:
```
@inproceedings{APICoder,
  title={When Languange Model Meets Private Library},
  author={Zan, Daoguang and Chen, Bei and Lin, Zeqi and Guan, Bei and Wang, Yongji and Lou, Jian-Guang},
  booktitle={EMNLP findings},
  year={2022}
}
```
