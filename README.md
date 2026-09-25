# Arctic retrievals for the Arctic Weather Satellite

**Author:** Eleanor May

This repository contains scripts to train, perform, and analyse retrievals of frozen water path (FWP), liquid water path (LWP), mean mass height (Zm) and mean mass diameter (Dm) from Arctic Weather Satellite (AWS) observations. The model used is a quantile regression neural network (QRNN). This work acts as an extension of CHIP-AWS.

## Setup

Requires mamba. From the repository root:

```bash
mamba env create -f environment.yml
mamba activate aws_arctic
```


## Repository structure

```
├── environment.yml
├── figures
│   └── swath_examples
├── plotstyling.mplstyle
├── README.md
└── scripts
    └── batch_plot_l2.py        Plot retrievals and Ta of single swaths
```


## Description of scripts

### Plotting scripts:
1. **`batch_plot_l2.py`** Plot retrieved variables or antenna temperatures from L2 Arctic files. Used to batch plot a large number of swaths, e.g. over a whole day or month. All data is plotted as a swath on Polar North maps.
