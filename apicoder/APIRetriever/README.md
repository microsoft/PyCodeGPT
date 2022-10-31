# What is it?
APIRetriever is a dense retrieval system that can find possible used APIs for programming problem. We refer to a toolkit named [Dense](https://github.com/luyug/Dense) to implemente our APIRetriever.

---

## Installation
Our dependencies are as follows:
```
pytorch==1.8.0
faiss-cpu==1.6.5
transformers==4.2.0
datasets==1.1.3
wandb==0.13.3
```
So, you should run pip commands for installing the above dependencies automatically.
```
cd Your/Own/Path/.../PrivateLibrary/APIRetriever
pip install .
```
Besides, if you would like to use mixed precision FP16 to speed up the training, it is necessary for you to install the apex library.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

## Project Directory
```shell
├── apex
├── data
│   ├── inference # The test data for five libraries. The below `XXX` can be `pandas`, `numpy`, `monkey`, `beatnum`, and `torchdata`.
│   │   ├── XXX_api.json # API and its id.
│   │   ├── XXX_api.pt # API embeddings encoded by our APIRetriever.
│   │   ├── XXX_comment.json # Code comment and its id.
│   │   ├── XXX_comment.pt # Code comment embeddings encoded by our APIRetriever.
│   │   ├── XXX_id_score.trec # The score between comment and API with an easy-to-read format.
│   │   ├── XXX_id_score.txt # The score between comment and API with an obscure format.
│   └── train
│       ├── processed-train-data
│       └── unprocessed-train-data
├── outputs
├── requirements.txt
├── scripts
│   ├── extract_retrieval_api_corpus.py
│   ├── run_extract_apiretriever_corpus.sh
│   ├── run_prepare_test_private_code.py
│   └── run_prepare_train_private_code.py
├── setup.py
└── src
    ├── dense
    ├── run_train_1.sh
    ├── run_encode_2.sh
    ├── run_search_3.sh
    └── run_trec_format_4.sh
```

## Training

First, you need to process the crawled python files into comment-API pairs.
```shell
bash APIRetriever/scripts/run_extract_apiretriever_corpus.sh
```
Then, you should convert these data pairs into a trainable format for training our APIRetriever.
```shell
python APIRetriever/scripts/run_prepare_train_private_code.py
```
After preparing the training corpus, you should start training your own APIRetriever.
```shell
bash APIRetriever/src/run_train_1.sh
```

<img src=https://s3.bmp.ovh/imgs/2022/09/27/fa49d48cba8f7860.png width=850 />

## Inference
After training phase, we can use APIRetriever to retrieve private APIs for each programming problem description. In detail, we apply $E_{\mathbf{a}}$ to all the APIs and index them by [FAISS](https://github.com/facebookresearch/faiss) offline. Given a new programming problem description $\mathbf{p}$ at run-time, we only need to produce its embedding $v_{\mathbf{p}}=E_{\mathbf{p}}(\mathbf{p})$ and recall the top-$k$ APIs with the embeddings closest to $v_{\mathbf{p}}$.

First, you should encode the code comments and APIs.
```shell
bash APIRetriever/src/run_encode_2.sh
```
Then, you need to retrieve and rank the APIs for each code comment.
```shell
bash APIRetriever/src/run_search_3.sh
```
Next, you can get the final scores between code comments and its APIs.
```shell
bash APIRetriever/src/run_trec_format_4.sh
```

> The retrieved outcome is placed in `APIRetriever/data/inference`. In addition, they can be used to prompt APIs (Top-1,2,3,5 and Human) to our crafted benchmarks.

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
