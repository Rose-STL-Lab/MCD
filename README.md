# Discovering Mixtures of Structural Causal Models from Time Series Data

Implementation of the paper "Discovering Mixtures of Structural Causal Models from Time Series Data", set to appear at ICML 2024, Vienna.

## Requirements

- NVIDIA GPU with minimum CUDA 11.8 installed.
- Make sure you have `conda` installed.

## Setup

Create a conda environment and install the prerequisite packages:
```
conda create -n mcd python=3.9 -y && \
conda run --no-capture-output -n mcd pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
conda run --no-capture-output -n mcd pip3 install lightning matplotlib numpy scikit-learn seaborn \
    cdt wandb igraph pyro-ppl hydra-core yahoofinancials && \
```

You also need the `graphviz` library. This library can be installed on Ubuntu systems using the command:
```
sudo apt-get install -y git && \
sudo apt-get install -y graphviz graphviz-dev
```

For baselines:
```
conda create -n baselines python=3.8 -y && \
conda run --no-capture-output -n baselines pip3 install pygraphviz wandb tigramite hydra-core pyro-ppl lightning causalnex matplotlib cdt seaborn lingam
```


## Dataset generation

Generate the datasets using the script files in the `scripts/` folder.

- Linear synthetic dataset: Run `./scripts/generate_synthetic_datasets_linear.sh`
- Nonlinear synthetic dataset: Run `./scripts/generate_synthetic_datasets_nonlinear.sh`
- Netsim datasets: Run `./scripts/setup_netsim.sh`
- DREAM3: Run `./scripts/setup_dream3.sh`
- S&P100: Run `./scripts/generate_snp100.sh`

## Running the code

Change the name of the `wandb` project in the config file.

- Linear synthetic dataset: Run `python3 -m src.train +dataset=ER_ER_num_graphs_<K>_lag_2_dim_<D>_NoHistDep_0.5_linear_gaussian_con_1_seed_0_n_samples_1000 +synthetic=mcd_linear`. Change `<D>` and `<K>` to the correct setting.  
- Nonlinear synthetic dataset: Run `python3 -m src.train +dataset=ER_ER_num_graphs_<K>_lag_2_dim_<D>_HistDep_0.5_mlp_spline_product_con_1_seed_0_n_samples_1000 +synthetic=mcd`. Change `<D>` and `<K>` to the correct setting.  
- Netsim-mixture: Run `python3 -m src.train +dataset=netsim_15_200_permuted +netsim=mcd`
- DREAM3: Run `python3 -m src.train +dataset=dream3 +dream3=mcd`
- S&P100: Run `python3 -m src.train +dataset=snp100 +snp100=mcd`

Results are stored in the `results/` folder.

## Acknowledgement

We borrowed a lot of code from [Project Causica](https://github.com/microsoft/causica).

## Citation

If you find this work useful, please consider citing us.

```
@inproceedings{varambally2024discovering,
  author       = {Varambally, Sumanth and Ma, Yi-An and Yu, Rose},
  title        = {Discovering Mixtures of Structural Causal Models from Time Series Data},
  booktitle    = {International Conference on Machine Learning, {ICML} 2024},
  series       = {Proceedings of Machine Learning Research},
  year         = {2024}
}
```
