<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/trust/blob/main/trust_logo.svg" width="500" height="150"/>
    </br>
</p>

<p align="center">
    <a href="https://github.com/decile-team/trust/blob/main/LICENSE">
        <img alt="GitHub license" src="https://img.shields.io/github/license/decile-team/trust"></a>
    </a>
    <a href="https://decile.org/">
        <img alt="Decile" src="https://img.shields.io/badge/website-online-green">
    </a>
    <a href="https://trust.readthedocs.io/en/latest/index.html">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-passing-brightgreen">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/decile-team/trust">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/decile-team/trust">
    </a>
    <a href="https://github.com/decile-team/trust/issues">
        <img alt="GitHub issues" src="https://img.shields.io/github/issues/decile-team/trust">
    </a>
</p>

# About TRUST

<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/trust/blob/main/tss.png" width="763" height="417"/>
    </br>
</p>

Efficiently search and mine for specific (targeted) classes/slices in your dataset to improve model performance and personalize your models.
TRUST supports a number of algorithms for targeted selection which provides a mechanism to include additional information via data to priortize the semantics of the selection.

## Starting with TRUST

### From Git Repository
```
git clone https://github.com/decile-team/trust.git
cd trust
pip install -r requirements/requirements.txt
```

## Where can TRUST be used?
TRUST is a toolkit which provides support for various targeted selection algorithms. Most real-world datasets have one or more charateristics that make its use on the state-of-the-art subset selection algorithms very difficult. Quite often, these characteristics are either known or can be easily found out. For example, real-world data is imbalanced, redudant and has samples that are of not of concern to the task at hand. Hence, there is a need to favor some samples while ignore the others. This is possible via different Submodular Information Measures based algorithms implemented in TRUST.

## Package Requirements
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "submodlib"


## Documentation
Learn more about TRUST by reading our [documentation](https://trust.readthedocs.io/en/latest/index.html).

## Tutorials
1. [Rare Classes Demo](https://colab.research.google.com/drive/1iidYqUu2Vkv_9lbIwvuwWKYPkhHt-vHR?usp=sharing)
2. [Fairness Demo](https://colab.research.google.com/drive/1STgb2cBzKPmXMChlq5Zv2VMhLHqUssyr?usp=sharing)


You can also download the .ipynb files from the tutorials folder.

## Acknowledgment
This library takes inspiration, builds upon, and uses pieces of code from several open source codebases. This includes [Submodlib](https://github.com/decile-team/submodlib) for submodular optimization.

## Team
TRUST is created and maintained by [Suraj Kothawade](https://personal.utdallas.edu/~snk170001/), Nathan Beck, and [Rishabh Iyer](https://www.rishiyer.com). We look forward to have TRUST more community driven. Please use it and contribute to it for your research, and feel free to use it for your commercial projects. We will add the major contributors here.

## Publications

[1] Kothawade S, Kaushal V, Ramakrishnan G, Bilmes J, Iyer R. PRISM: A Rich Class of Parameterized Submodular Information Measures for Guided Subset Selection. To Appear In 36th AAAI Conference on Artificial Intelligence, AAAI 2022

[2] Iyer, R., Khargoankar, N., Bilmes, J. and Asanani, H., 2021, March. Submodular combinatorial information measures with applications in machine learning. In Algorithmic Learning Theory (pp. 722-754). PMLR.

[3] Anupam Gupta and Roie Levin. The online submodular cover problem. In ACM-SIAM Symposiumon Discrete Algorithms, 2020
