[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Godofnothing/graf/blob/main/main.ipynb)

### **The project with high GPU demand**

# GRAF reimplemetation

<div style="text-align: center">
<img src="animations/results.gif" width="1024"/><br>
</div>

This repository contains the reimplemetation of the official code for the paper
[GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://avg.is.tuebingen.mpg.de/publications/schwarz2020neurips).

You can find detailed usage instructions for using pre-trained models and training your own models below.

## Usage

1) Click on the button `Open in Colab` or go to the `main.ipynb` and have a look on the notebook;
2) Set **GPU** as a runtime (**NOTE:** it's very crucial to choose the GPU runtime); 
3) Follow the instructions: choose one of the datasets and choose one of the options of the transfer learning;
4) **Run All cells**;
5) Choose `.json` file for **kaggle**; [(You may find how to download JSON file here)](https://github.com/Kaggle/kaggle-api#:~:text=To%20use%20the%20Kaggle%20API,file%20containing%20your%20API%20credentials.)
6) After starting the training process you should wait for `~15-20 minutes` and then the folder `results/NAME_OF_CURRENT_FOLDER` should be created, where you can find generated images and videos varying camera pose chosen datasets;
7) After you've decided to stop, the iterations go to the next cell and save your results locally;
8) Download the `stats.py` file from the `result` folder; 
9) Open `plot_stats.ipynb` to plot the results on `FID` and `KID`.

## Transfer learning on your own dataset

1) Set-up the config file, look at the examplle `configs/transfer_learning_ffhq_freezed_but_last.yaml`. Then consider next things:
- Learning rate of the generator can be a `float` or `dict` (where keys are the names of modules);
- Learning rate of the discriminator can be a `float` or `list`; 
- In both cases check, that the length of learning rate list matches the number of layers;
- Image sizes of the dataset, on which the generator is trained, and from which we transfer the weights *have to be equal*;
- Don't forget to set the names of the initial dataset and the target dataset in config file.
2) When running the `python train.py` add a flag `--pretrained` in order to run the model with pretrained weights.
3) Note: while running in `Colab` you should run the main code within one cell. 

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

**Note:** use this useful code for installing conda in Colab:
```
import sys
# Download miniconda
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local
# Install packages from Anaconda
!conda install -y -c conda-forge --prefix /usr/local pymeep
# Append path to be able to run packages installed with conda
sys.path.append('/usr/local/lib/python3.7/site-packages')
```

You can create an anaconda environment called `graf` using
```
conda env create -f environment.yml
conda activate graf
```

Next, for nerf-pytorch install torchsearchsorted. Note that this requires `torch>=1.4.0` and `CUDA >= v10.1`.
You can install torchsearchsorted via
``` 
cd submodules/nerf_pytorch
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ../../../
```

## Datasets

**The pre-trained** models were trained on **CelebFaces Attributes Dataset(CelebA)**, **Carla Dataset**, and **Cat Dataset** datasets:
- [CelebFaces Attributes Dataset (CelebA)](https://www.kaggle.com/jessicali9530/celeba-dataset)
- [Carla Dataset](https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip)
- [Cat Dataset](https://www.kaggle.com/crawford/cat-dataset)

We had to decide what type of datasets to use for conducting our experiments. We wanted to choose at least one ideal dataset (**Anime**), two potentially good datasets (**FFHQ** and **Dogs**), similar to the basic ones, and another with highly different types of data (**Fruits**). 

**The target** models were trained on the next datasets: 
- [Flickr-Faces-HQ Dataset (FFHQ)](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
- [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset)
- [Fruits 360](https://www.kaggle.com/moltean/fruits)
- [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)

The target models were trained using  base models in the next way:

1) CelebA 🠒 FFHQ
2) CelebA 🠒 Anime 
3) Cats 🠒 Dogs 
4) Carla 🠒 Fruits 

**Note**: _base dataset_ 🠒 _target dataset_.

Due to computational restrictions, we've used the next sizes of the target datasets:
- FFHQ: 10k images;
- Anime Face: 63k images (full dataset);
- Stanford Dogs: 3.5k images;
- Fruits: 600 images.

### Stanford dogs

In this kind of dataset, we considered some manual settings to choose the best samples with the lowest level of the background.

### Fruits 360

In the case of this kind of dataset, we've considered manual settings to avoid bad results on different types of fruits (It is possible, but it cost a lot of computational capacity). We've decided to have a test with all kinds of apple (and with just one type - in this way we achieved much better results).

## Train a model from scratch

Based on our experiments (as you may see Anime is the best one), we suggest you use a larger dataset (e.g., at least 30 000 images).

To train a 3D-aware generative model from scratch run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with your config file.
The easiest way is to use one of the existing config files in the `./configs` directory 
which correspond to the experiments presented in the paper. 
Note that this will train the model from scratch and will not resume training for a pretrained model.

**Note:** to train a model from scratch, you should consider a new `CONFIG.yaml` file based on `default.yaml`!

## Evaluation of a new model

For evaluation of the models run
```
python eval.py CONFIG.yaml --fid_kid --rotation_elevation --shape_appearance
```
where you replace `CONFIG.yaml` with your config file.

## Further Information

### GRAF

We're very thankful to the [GRAF repository](https://github.com/autonomousvision/graf).

### GAN training

GRAF repository uses Lars Mescheder's awesome framework for [GAN training](https://github.com/LMescheder/GAN_stability).

### NeRF

The GRAF repository code is based on the Generator on this great [Pytorch reimplementation](https://github.com/yenchenlin/nerf-pytorch) of Neural Radiance Fields.

### Some hints

If you suffer from lack of memory set batch size as small as possible - like 1 in `configs/default.yml`.


