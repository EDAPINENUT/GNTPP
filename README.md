# Exploring Generative Temporal Point Process


This is a PyTorch implementation of GNTPP in the paper `Exploring Generative Neural Temporal Point Process' (TMLR). 


### Requirements
* torch>=1.7.0 
* ticks
* xarray
* torchdiffeq

Dependency can be installed using the following command:
```bash
conda env create --file env_gntpp.yaml
conda activate GNTPP_env
```
### Training the Model

Run the following commands to train the model.

```bash
#  Training the Model
python tpp_train.py --dataset_dir ./data/{dataset}/ --hist_enc {encoder} --prob_dec {decoder}
```

{dataset} can be replaced by `[mooc, retweet, stackoverflow, yelp, synthetic_n5_c0.2]`.

{encoder} can be replaced by `[LSTM, Attention]`.

{decoder} can be replaced by `[CNF, Diffusion, GAN, ScoreMatch, VAE, LogNorm, Gompt, Gaussian, Weibull, FNN, THP, SAHP]`.

(NOTE: The provided THP and SAHP use different type-modeling methods (type-wise intensity modelling), while others model all the types in a single sequence. So the final metric evaluation will be in a different protocol.)

Some of the datasets are larger than allowed, so we provided all these datasets file at [Google Drive](https://drive.google.com/drive/folders/1yQ3BB4S3twL4i_VbMUikANbmBMI0aNR7?usp=sharing). 

### Preparing Your Own Datasets
If you want to use your own dataset, please prepare it in a `Dict`, where the keys include `timestamps`, `types`, `lengths`, `intervals`, `t_max` and `event_type_num`. (Refer to Line.88 in `./datasets/tpp_loader.py`).

### Building up Your Own Neural TPP
If you want to build up your own model, please refer to `./test.py`, to see how the different modules in our paper constitute the TPP models.

### Relation Visualization
For learned model, `Trainer.plot_similarity` method could provide the relations among events learned by the model. A similarity matrix will be generated, which looks like the following:

<p align="center">
  <img src='./figs/type_similarity_sof.png?raw=true' width="400">
</p>

If the repository is helpful to your research, please cite the following:

```
@article{
lin2022GNTPP,
title={Exploring Generative Neural Temporal Point Process},
author={Haitao Lin and Lirong Wu and Guojiang Zhao and Pai Liu and Stan Z. Li},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=NPfS5N3jbL},
note={}
}
```

or

```
@misc{lin2022GNTPP,
  doi = {10.48550/ARXIV.2208.01874},
  
  url = {https://arxiv.org/abs/2208.01874},
  
  author={Haitao Lin and Lirong Wu and Guojiang Zhao and Pai Liu and Stan Z. Li},

  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Exploring Generative Neural Temporal Point Process},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
