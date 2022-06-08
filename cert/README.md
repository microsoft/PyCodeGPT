# CERT: Continual Pre-training on Sketches for Library-oriented Code Generation

CERT's source code and our crafted evaluation benchmarks.

## Installation

### For benchmarks installation
```
$ unzip human-eval.zip
$ pip install -e human-eval
```

### For the installation of the CERT runtime environment
```
$ pip install -r requirements.txt
```

## Usage

### Encoding the cleaned code corpus: 

- Converting each code file to many code blocks
- Each code block is converted to code sketch
- Tokenizing code and converting text to binary file.

```
$ bash scripts/run_encode_domain.sh
```

### Training CERT
```
$ bash run_cert.sh
```

### Evaluating CERT

Our crafted PandasEval and NumpyEval are placed in human-eval/data.

```
$ bash run_eval_monitor.sh
```

Assign the output file path from the previous step to the POST_PATH variable in run_eval_monitor_step2.sh.

```
$ bash run_eval_monitor_step2.sh
```


## Citation

Please cite using the following bibtex entry:

```
@inproceedings{CERT,
  title={{CERT}: Continual Pre-training on Sketches for Library-oriented Code Generation},
  author={Zan, Daoguang and Chen, Bei and Yang, Dejian and Lin, Zeqi and Kim, Minsu and Guan, Bei and Wang, Yongji and Chen, Weizhu and Lou, Jian-Guang},
  booktitle={The 2022 International Joint Conference on Artificial Intelligence},
  year={2022}
}
```