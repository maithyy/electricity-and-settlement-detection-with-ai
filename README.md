[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)
# kMM Classifier: Settlement & Electricity Detection
_**Detection of Settlements via Random Forest Classifier & Electricity via Max Projection VIIRS Algorithm**_

### Team Members
- **K**atelyn Wang ([@katelyn-wang](https://github.com/katelyn-wang))
- **M**ichael Pien ([@ThatMegamind](https://github.com/ThatMegamind))
- **M**aithy Le ([@maithyy](https://github.com/maithyy))

## Links
[[`Tech Memo`](https://docs.google.com/document/d/1A6dgTovOp_WDhmoUxQ9LbWnWVLgJbqUmYb9XirHIXCI/edit?usp=sharing)]
[[`Presentation Slides`](https://docs.google.com/presentation/d/1B8fDvT2_o-qfTbQsTG-V65KCur3RKCPd1jsFepa6k8o/edit?usp=sharing)]
[[`Presentation Recording`](https://youtu.be/eBG-9rsoqnA)]
[[`Project Poster`](https://docs.google.com/presentation/d/1kHa3TOPUpb5FEkLUM7esWrIA2djC7gMBXTiMt8S4M_k/edit?usp=sharing)]

##
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-7D00FF?style=for-the-badge&logo=PyTorch_Lightning&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Xarray](https://img.shields.io/badge/Xarray-48B9C7?style=for-the-badge&logo=Xarray&logoColor=white)
![Wandb](https://img.shields.io/badge/Wandb-F9DC3e?style=for-the-badge&logo=Wandb&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Our Task
1. Use binary segmentation to identify **settlements** and **non-settlements** with a random forest classifier
  
2. Classify settlements as having **electricity** or **no electricity** using VIIRS nighttime data

## Overview of Project Architecture

The architecture of our project can be split into 3 distinct phases: 
1. **Preprocessing**
2. **Settlement Detection**
3. **Electricity Detection**
   
In our **preprocessing** step, we run a series of functions which extract, cleanup, and prepare the original satellite images to be fed into our machine learning models. 

In the **settlement detection** phase, we can run one of four models: Random Forest Classifier (primary model), Segmentation CNN, Transfer Resnet101, or U-Net. These models produce a classification of pixels as either settlement or non-settlement. 

Finally, in the **electricity detection** phase, we run an algorithm that takes the predictions created by the machine learning model and uses the max VIIRS projection data to determine whether each predicted settlement has electricity or not, producing a final output that identifies settlements with no electricity.


## Pipeline

![CS 175 Final Pipeline (4)](https://github.com/cs175cv-s2024/final-project-kmm-classifier/assets/92563733/f80e1dd7-0d9b-4ba8-ad0e-4d57c708b6b8)

## Segmentation Sample/Result

![Combined Results](https://github.com/cs175cv-s2024/final-project-kmm-classifier/assets/60128757/20cfa69a-360e-4c9f-8c2a-a8a80c1460c3)

## Installation + Getting Started

To begin, start by cloning this repository by running

```
git clone https://github.com/cs175cv-s2024/final-project-kmm-classifier.git
```

Then, navigate to the project folder and run the following command to install the necessary libraries and packages

```
pip install -r requirements.txt
```

This will successfuly set up the repository for use. 

Next, download the necessary data from the [2021 IEEE GRSS Dataset](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view), and place the Train data into a directory titled `data/raw/Train`. 

To prepare the data for the binary segmentation task, make sure to run `scripts/extra_utils.py` before anything else.

Now, you can train any of the models provided by setting flags in the command to run the program. Notably, you can set the model of Random Forest Classifier by setting the flag `--model_type: RandomForestClassifier`. With the model_type flag plus any additional desired parameters, run `python3 scripts/train.py`. This should produce a `.ckpt` file under the models directory. 

You can then run `python3 scripts/predict_electricity.py` to produce a final prediction based on the output of the trained model, indicating what areas are likely to have settlements without electricity.


## How We Assembled Our Architecture
#### Data Preprocessing
- [X] Created new ground truth images, where 1 color is for settlements with electricity + settlements w/o electricity, and another color is the other two classes

#### Random Forest Classifier
- [X] Created a new file called `random_forest_module.py` inside `models/supervised`, which contains our PyTorch implementation of random forest model
  - Used for classifying settlements vs. non-settlements
- [X] Modified `satellite_module.py` to account for new type of model
- [X] Modified `utilities.py` to account for new hyperparameters

#### Electricity Classification
- [X] Wrote an algorithm to take the ouput of the random forest classifier to classify electricity vs. no electricity
  - Used VIIRS nighttime dataset and `maxprojection_VIIRS`
- [X] Calculated thresholds for what qualifies as having electricity

#### Combining the Model and Algorithm
- [X] Created another set of ground truth images, where 1 color is for settlements with no electricity and the other is for everything else
- [X] Inputted the settlement detection output into the electricity detection algorithm to get final prediction of settlements with no electricity


## Citing kMM Classifier
If you use kMM Classifier's work in your research, please use the following BibTeX entry.

```
@article{wang_le_pien_2024_settlement_detection,
  title={kMM Classifier Settlement and Electricity Detection},
  author={Wang, Katelyn and Le, Maithy and Pien, Michael},
  year={2024}
}
```

## Reference Papers
- [Mapping New Informal Settlements Using Machine Learning and Time Series Satellite Images](https://ieeexplore.ieee.org/abstract/document/9311041)
- [Detection of rural electrification in Africa using DMSP-OLS night lights imagery](https://www.tandfonline.com/doi/full/10.1080/01431161.2013.833358)
