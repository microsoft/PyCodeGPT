# APICoder - CodeGenAPI

Official repository for our paper ["When Language Model Meets Private Library"](https://arxiv.org/pdf/2210.17236.pdf).

---

## Overview

APIRetirever finds out useful APIs for a programming problem, and then APICoder aims to generate code that solves the problem with these APIs. We make use of the most straightforward way for APICoder: prompting API information set in front of the context. Each API information is in the form of `name(signature):description`. This is to mimic programmers learning the APIs properly before writing code using them.

<img src=https://s3.bmp.ovh/imgs/2022/09/27/3691aaf9d0421991.png width=650 />

Figure1: The training process of CodeGenAPI

## Project Directory
```shell
├── CodeGenAPI
│   ├── APICoder
│   │   ├── get_api_info_by_name.py
│   │   ├── get_lib_comment_for_eval.py
│   ├── apex
│   ├── eval_baseline.py
│   ├── eval_private.py
│   ├── nl2code
│   ├── requirements.txt
│   ├── run_generating_codes.sh # The entry script for CodeGenAPI inference, which can generate a lot of code snippets for each programming problem.
│   ├── run_evaluating_codes.sh # The entry script for evaluating the generated code snippets, and outputting the final results (pass@k).
│   ├── run_private.py
│   ├── run_private.sh # Implementation of CodeGenAPI training.
│   └── scripts
│       ├── encode_private_data.py
│       ├── extract_api.py
│       ├── file_utils.py
│       ├── get_comments_from_evallibs.py
│       ├── get_libs_info_from_code.py
│       ├── make_human_in_the_loop_test_corpus.py
│       ├── multiprocessing_utils.py
│       ├── pycode_visitor.py
│       ├── requirements.txt
│       ├── run_details_apis.sh # Extracting all kinds of API information (API name, signature, description and so on) from the crawled API documentations of 35 libraries.
│       ├── run_encode_private_data.sh # Encoding the private data
│       ├── run_extract_apis.sh # Crawling the API documentation for 31 off-the-shelf public libraries.
│       └── run_extract_details_from_apis.py
```

## Quickstart

This section covers environment, data preparation, model inference, and model training.

### Preparation

1、Configuring your runtime environment

```
$ cd PrivateLibrary/CodeGenAPI
$ pip install -r requirements.txt
```
Besides, if you would like to use mixed precision FP16 to speed up the training, it is necessary for you to install the apex library.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

2、Preparation of pre-trained models

Download the pre-trained checkpoint (e.g., `CodeGenAPI-110M`) from [our released page](https://github.com/microsoft/PyCodeGPT/releases/download/Private-Library/CodeGenAPI-350M-mono.zip) and place it in the corresponding folder (e.g., `CodeGenAPI/models/CodeGenAPI-110M`).

3、Updating the scripts according to your local path

- Update `run_private.sh`.
- Update `run_generating_codes.sh`.
- Update `run_evaluating_codes.sh`.

### Use CodeGenAPI or other models

Firstly, multiple code snippets are generated for each programming problem (`run_generating_codes.sh`). Then, the code snippets are evaluated (`run_evaluating_codes.sh`).

```
$ bash run_generating_codes.sh
$ bash run_evaluating_codes.sh
```

### Train CodeGenAPI

Train CodeGenAPI by the following command based on the large-scale code corpus.

```
$ bash run_private.sh
```

## Experiments

In inference phase, we set the `temperature` to one of `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`, the number of samples (`NUM_SAMPLES`) to `200`, the max number of generated tokens (`MAX_TOKNES`) to `100`, and the `top_p` to `0.9`. The best number is reported across the above hyper-parameters.

Here are the main results:

![](https://s3.bmp.ovh/imgs/2022/09/27/1f28c06f5cc05bcc.png)

After running these numerous experiments, we drew some plausible observations and valuable insights as follows.

> (1) Prompting API information set is useful on private-library oriented code generation task.

> (2) Which is the best of the API prompt ways including Perfect, Top-N, and Human? As a general matter, Perfect, Human, and Top-N produce progressively decreasing benefits. However, Top-N is in occasion superior than Perfect as the noise exists when training the model. Also, we observe that Top-1,2 usually works better than Top-3,5 because the latter introduces more noise APIs. 

> (3) Our continual pre-trained model does better at invoking APIs than to its base model, and thus can further elevate the performance of code generation for private libraries in majority of scenarios.

> (4) APIRetriever has the capability to retrieve useful APIs.

> (5) Involving human in the loop can further boost the performance.

> (6) As the k in pass@k grows larger, the gain we add API information brings is larger.

> (7) It is so challenging to generate code invoking private libraries than public ones, that large models fail to do so if we do not prompt any APIs.

For more explanation, please see our raw paper.

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
