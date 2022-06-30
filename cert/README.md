# CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation

Official repository for our paper ["CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation"](https://arxiv.org/pdf/2206.06888.pdf), containing crafted benchmarks, codes, and pre-trained models.

---

## Overview

In our paper, we focus on investigating whether and how language models pre-trained on large-scale unlabelled code corpus can generate library-oriented code snippets. To meet this challenge, we propose CERT (for sket**C**her and g**E**ne**R**a**T**or), which is a continual pre-training approach on sketches for library-oriented code generation. In CERT, a sketcher firstly focuses on predicting a sketch, which omits user-defined details; then, a generator uses the sketch as a prompt to generate the complete code. In addition, we craft two evaluation benchmarks for Python libraries, called PandasEval and NumpyEval, each including 101 programming problems using Pandas and NumPy, respectively.

<img src=https://s3.bmp.ovh/imgs/2022/06/28/98ea5bdc5d86fbc8.png width=450 />

Figure1: Overview of CERT: a sketcher and a generator.

## Project Directory
```
├── nl2code # Basic scripts for loading corpus and training CERT.
    ├── code_dataset.py
    ├── dynamic_block_dataset.py
    ├── hf_trainer.py
    └── indexed_dataset.py
├── pandas-numpy-eval # Benchmarks and evaluation scripts. Please go to the folder for details.
├── scripts
    ├── ast_utils.py # Tools to handle the AST of Python code, for example, converting a code block to its code sketch.
    ├── encode_domain.py # Implementation of encoding.
    ├── file_utils.py # Tools for managing files.
    ├── multiprocessing_utils.py # Tools for managing multiple processes.
    └── run_encode_domain.sh # Encoding the crafted corpus (sketcher corpus and generator corpus).
├── eval_cert_unified.py # Implementation of code generation for CERT-sketcher and CERT-generator.
├── eval_cert.py # Implementation of code generation for PyCodeGPT and other baseline models.
├── run_cert.py # Implementation of CERT training.
├── run_evaluating_codes.sh # The entry script for evaluating the generated code snippets, and outputting the final results (pass@k).
├── run_generating_codes.sh # The entry script for CERT inference, which can generate a lot of code snippets for each programming problem in PandasEval and NumpyEval.
├── run_training_cert.sh # The entry script for training CERT.
```

## Quickstart

This section covers environment, data preparation, model inference, and model training.

### Preparation

1、Configuring your runtime environment

```
$ cd CERT/
$ pip install -r requirements.txt
```

2、Preparation of pre-trained models

Download the pre-trained checkpoint (e.g., `pycodegpt-110M`) from `Releases` in this GitHub project and place it in the corresponding folder (e.g., `CERT/models/pycodegpt-110M`).

3、Updating the scripts according to your local path

- Update `run_training_cert.sh`.
- Update `run_generating_codes.sh`.
- Update `run_evaluating_codes.sh`.

### Use PyCodeGPT or CERT

Firstly, multiple code snippets are generated for each programming problem (`run_generating_codes.sh`). Then, the code snippets are evaluated (`run_evaluating_codes.sh`).

```
$ bash run_generating_codes.sh
$ bash run_evaluating_codes.sh
```

### Train CERT

Train CERT (sketcher and generator) by the following command based on the large-scale code corpus.

```
$ bash run_training_cert.sh
```

## Experiments and Some Cases

In inference phase, we set the `temperature` to one of `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`, the number of samples (`NUM_SAMPLES`) to `200`, the max number of generated tokens (`MAX_TOKNES`) to `100`, and the `top_p` to `0.9`. The best number is reported across the above hyper-parameters.

Here are some cases:

1. Sketcher and Generator are able to predict successfully. (It usually occurs when there are more user-defined terms.)

2. Sketcher predicts the correct answer directly. (It usually occurs when there are relatively few or no user-defined terms)

3. Sketcher predicts wrong sketch, but generator can rectify them and predict the correct answer.

<img src=https://s3.bmp.ovh/imgs/2022/06/29/34fb125ffcc23758.png width=900 />

## Citation
If you find our work useful, please cite the paper:
```
@inproceedings{CERT,
  title={{CERT}: Continual Pre-training on Sketches for Library-oriented Code Generation},
  author={Zan, Daoguang and Chen, Bei and Yang, Dejian and Lin, Zeqi and Kim, Minsu and Guan, Bei and Wang, Yongji and Chen, Weizhu and Lou, Jian-Guang},
  booktitle={The 2022 International Joint Conference on Artificial Intelligence},
  year={2022}
}
```