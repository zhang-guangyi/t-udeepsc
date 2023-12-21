# PyTorch implementation of [A Unified Multi-Task Semantic Communication System for Multimodal Data](https://arxiv.org/abs/2209.07689)

This repository is built upon [BEiT](https://github.com/microsoft/unilm/tree/master/beit) and [MAE](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!

We would gradually upload the full-version of the implementation.

## DataSet Preparation

## TODO
- [x] implement the designed benchmark: task-oriented semantic communication works (T-DeepSC) for the considered tasks, including image(cls/recons), text(cls/recons) under analog transmission.
- [x] implement the designed benchmark: TDeepSC for VQA and MSA under analog transmission.
- [x] implement the digital transmission: vector quantization (VQ) and uniform scalar quantization (SQ). 
- [x] implement the 16QAM and QPSK modulations.
- [ ] dataset preparation
- [X] the basic version of the unified semantic communication (U-DeepSC).




## Run
1. The instructions are given in execute.sh, use "bash execute.sh" to execute the script.
2. All instructions are given in running_command.sh.
