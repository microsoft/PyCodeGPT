# TorchDataEval, MonkeyEval and BeatNumEval

Three benchmarks for evaluating the performance of private library oriented code generation. They are proposed in the paper "[When Language Model Meets Private Library](https://arxiv.org/pdf/2210.17236.pdf)". 

The evaluation script is adapted from OpenAI's [HumanEval](https://github.com/openai/human-eval/tree/master/human_eval).

## Installation

Make sure to use python 3.7 or later: 
```
$ conda create -n private python=3.7
$ conda activate private
```

Check out and install this repository:
```
$ pip install -e private-eval
```

## Configuration
```
├── data # The directory of our crafted benchmarks.
├── private_eval
│   ├── data.py # [key] Choosing whether to load TorchDataEval, MonkeyEval or BeatNumEval.
│   ├── evaluate_functional_correctness.py # Calculating the evaluation results.
│   ├── evaluation.py # Calculating the evaluation results.
│   └── execution.py # [key] Executing the predicted code. Here, if you want to evaluate MonkeyEval and BeatNumEval, you need to set the `is_convert_back` variable in line 194 to `True` and `domain` to `pandas` or `numpy`.
```

## Running Environment Testing

You need replace `XXX` with your local path for testing the torchdata results. (Make sure that all settings in `private-eval/private_eval/data.py` is right.)
```
$ evaluate_functional_correctness XXX/PrivateLibrary/private-eval/data/TorchData_no.API_number_0.CodeGen.hm_False.machine.t0.1.p0.9.l100.n1.samples.jsonl
```

If you can successfully run the above command and obtain the following results, the evaluation environment is ready to use.
```
{'pass@1': 0.06}
```

# The Process of Constructing TorchDataEval, MonkeyEval and BeatNumEval

We craft three benchmarks, called TorchDataEval, MonkeyEval, and BeatNumEval. Each programming problem consists of context, target code, and the corresponding test cases.

To create a realistic benchmark for evaluating code generation for private library, we make use of TorchData, a Python library released just recently. We carefully learnt the official API documentation of TorchData and make sure we were proficient in all APIs. Then, we manually created $50$ programming problems based on the API usage examples in the documentation. Two volunteers with extensive experience in Python were invited to check the correctness of each problem. We control the difficulty of the programming problems by the number of APIs in the target code. The percentage of programming problems containing $1$ API, $2$ APIs, and more APIs is set to $6$:$3$:$1$.

> Our base model, CODEGEN, is pre-trained with GitHub data before $2021$-$10$. TorchData was released after this time point and no code files using it are available on GitHub so far, hence we can consider it as a private library.

We also construct two pseudo private libraries named MonkeyEval and BeatNumEval, they modify from PandasEval and NumpyEval, each containing $101$ programming problems, were proposed for the public libraries Pandas and Numpy. In detail, we manually modified all library-related keywords in PandasEval and NumpyEval, respectively. For example, as in the below Figure, `pandas` is converted to `monkey`, `dataframe` is converted to `knowledgeframe`, and the API name `isin` is converted to `iscontain`. To craft the API documentations for Monkey and BeatNum, we manually paraphrased the descriptions of all the new APIs to ensure that they have never been seen by the pre-trained language models.



# A Example of Converting PandasEval (public) to MonkeyEval (private)

Context is shown with a white background and the target code with a gray background. The changed parts are highlighted in yellow.

<img src=https://s3.bmp.ovh/imgs/2022/09/27/4c7196cf9a826984.png width=450 />

## Reference

If you use TorchDataEval, MonkeyEval or BeatNumEval in your work, please cite the paper:
```
@inproceedings{APICoder,
  title={When Languange Model Meets Private Library},
  author={Zan, Daoguang and Chen, Bei and Lin, Zeqi and Guan, Bei and Wang, Yongji and Lou, Jian-Guang},
  booktitle={EMNLP findings},
  year={2022}
}
```

If you use PandasEval or NumpyEval in your work, please cite the paper:
```
@inproceedings{CERT,
  title={{CERT}: Continual Pre-training on Sketches for Library-oriented Code Generation},
  author={Zan, Daoguang and Chen, Bei and Yang, Dejian and Lin, Zeqi and Kim, Minsu and Guan, Bei and Wang, Yongji and Chen, Weizhu and Lou, Jian-Guang},
  booktitle={The 2022 International Joint Conference on Artificial Intelligence},
  year={2022}
}
```

Also, if you use the evaluationg script, please also cite the following paper:
```
@article{codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
