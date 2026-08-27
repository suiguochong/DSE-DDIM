# Learning and Exploring the Hardware Dataflow Design Space through DDIM

This repository contains the official implementation of:
> **Learning and Exploring the Hardware Dataflow Design Space through DDIM**

The project proposes a **Denoising Diffusion Implicit Model (DDIM)** based framework for efficient hardware dataflow Design Space Exploration (DSE). Given a DNN workload and the number of processing elements (PEs), the proposed method generates high-performance dataflow configurations by learning the distribution of high-quality dataflows.

## Environment

The experiments were conducted with:

- Ubuntu 22.04
- Python 3.9.13
- PyTorch 2.5.0
- CUDA 12.4
- NVIDIA RTX A6000
- Intel Xeon Platinum 8488C

## Usage

### Reproduce the results of the paper
```sh
python methods_contrast.py --method DDIM --candidate_N 64
```
method: 'DDIM', 'DDPM', 'GA', 'GA_small', 'GA_large', 'MLP', 'Random'

### Retrain the DDIM
The training dataset can be downloaded from the repository release files.
```sh
python stage_2.py
```

## Citation

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{DSE-DDIM,
  title     = {Learning and Exploring the Hardware Dataflow Design Space through DDIM},
  author    = {XXX},
  booktitle = {PRICAI},
  year      = {2026}
}
```
