# PandasEval and NumpyEval

Two benchmarks for evaluating the performance of library-oriented code generation. They are proposed in the paper "[CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation](https://arxiv.org/pdf/2206.06888.pdf)". 

The evaluation script is adapted from OpenAI's [humaneval](https://github.com/openai/human-eval/tree/master/human_eval).

## Installation

Make sure to use python 3.7 or later: 
```
$ conda create -n pycodegpt python=3.7
$ conda activate pycodegpt
```

Check out and install this repository:
```
$ pip install -e pandas-numpy-eval
```

## Configuration
```
├── data # The directory of our crafted benchmarks.
│   ├── NumpyEval.jsonl.gz
│   └── PandasEval.jsonl.gz
├── pandas_numpy_eval
│   ├── data.py # Choosing whether to load PandasEval or NumpyEval.
│   ├── evaluate_functional_correctness.py # Calculating the evaluation results.
│   ├── evaluation.py # Calculating the evaluation results.
│   └── execution.py # Executing the predicted code.
```

## Running Environment Testing

You need replace `XXX` with your local path for testing the pandas results. (Make sure that the `LIB` variable in `pandas-numpy-eval/pandas_numpy_eval/data.py` is set to `pandas`.)
```
$ evaluate_functional_correctness XXX/CERT/pandas-numpy-eval/data/Example_Pandas_PYCODEGPT_samples.jsonl
```

If you can successfully run the above command and obtain the following results, the evaluation environment is ready to use.
```
{'pass@1': 0.06930693069306931}
```

# The Process of Constructing PandasEval and NumpyEval

We refer to [StackOverFlow](https://stackoverflow.com/), a Q&A website for programmers, to build the benchmarks. We search for posts using the library tag on StackOverFlow, and select those with high votes. To ensure quality, we only refer to posts with accepted answers. We go through a post's question and its accepted answer, then manually organize them into the form needed for our benchmarks, containing both context and target code. We also polish all programming problems so that the problem descriptions are clear and the codes are correct. Note that we keep the intentions and the descriptions of the programming problems consistent with the posts to the maximum extent. Finally, two programmers with more than three years of coding experience in the library are invited to act as code generation models and check the quality of the data.

As a result, we craft 101 programming problems for PandasEval and NumpyEval, respectively. Each programming problem is equipped with test cases for evaluation.

# Two Examples of Programming Problems

Context is shown with a white background and the target code with a gray background.

<!-- <img src=../images/benchmark_demo.png width=450 /> -->
<!-- ![](../images/benchmark_demo.png) -->
<img src=https://s3.bmp.ovh/imgs/2022/06/28/7a6fa5d9aa4ea6c7.png width=450 />

## Reference

If you use PandasEval or NumpyEval in your work, please cite the paper:

```
@inproceedings{CERT,
  title={{CERT}: Continual Pre-training on Sketches for Library-oriented Code Generation},
  author={Zan, Daoguang and Chen, Bei and Yang, Dejian and Lin, Zeqi and Kim, Minsu and Guan, Bei and Wang, Yongji and Chen, Weizhu and Lou, Jian-Guang},
  booktitle={The 2022 International Joint Conference on Artificial Intelligence},
  year={2022}
}
```

If you use the evaluationg script, please also cite the following paper:
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
