# PyTorch implementation of [A Unified Multi-Task Semantic Communication System for Multimodal Data](https://arxiv.org/abs/2209.07689)

This repository is built upon [BEiT](https://github.com/microsoft/unilm/tree/master/beit) and [MAE](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!

We would gradually upload the full-version of the implementation.

## Citation (Preprint Version)
``` bash
@ARTICLE{udeepsc,
	title={A Unified Multi-Task Semantic Communication System for Multimodal Data},
	author={Guangyi Zhang and Qiyu Hu and Zhijin Qin and Yunlong Cai and Guanding Yu and Xiaoming Tao},
	journal={arXiv preprint arXiv:2209.07689},
	year={2022}}
```


## Usage
### Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/zhang-guangyi/t-udeepsc.git
cd t-udeepsc/
```

### Requirements
`Python 3.8.5` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt (Not provided yet)
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend related documentation page](https://pytorch.org/get-started/previous-versions/).



## DataSet Preparation
### CIFAR10
Use the torchvision, the datasets will be dowmloaded automatically.

### MOSEI and MOSI




## TODO
- [x] implement the designed benchmark: task-oriented semantic communication works (T-DeepSC) for the considered tasks, including image(cls/recons), text(cls/recons) under analog transmission.
- [x] implement the designed benchmark: TDeepSC for VQA and MSA under analog transmission.
- [x] implement the digital transmission: vector quantization (VQ) and uniform scalar quantization (SQ). 
- [x] implement the 16QAM and QPSK modulations.
- [X] the basic version of the unified semantic communication (U-DeepSC).
- [x] dataset preparation.
- [x] feature selection-based UDeepSC.
- [ ] packages requirement.




## Run
1. The instructions are given in execute.sh, use "bash execute.sh" to execute the script.
2. All instructions are given in running_command.sh.
