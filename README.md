# BRSET: Brazilian Multilabel Ophthalmological Dataset Repository

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


Welcome to the GitHub repository for the Brazilian Multilabel Ophthalmological Dataset (BRSET). This repository is dedicated to the technical validation, data analysis, quality assessment, and modeling of the BRSET, which is a comprehensive dataset of retinal images labeled according to anatomical parameters, quality control, and presumed diagnosis. The BRSET is a valuable resource for the ophthalmology research community and aims to reduce under-represented countries in the dataset pool used for the development of models.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Quality Assessment](#quality-assessment)
- [Modeling](#modeling)
- [Citation](#citation)
- [License](#license)

## Introduction
The BRSET is the first Brazilian Multilabel Ophthalmological Dataset, comprising a total of 16,266 images from 8,524 patients, collected from 2010 to 2020. This dataset includes macula-centered JPEG retinal fundus photos captured using a Nikon NF505 and a Canon CR2 camera. In addition to retinal images, the dataset provides demographic information such as patients' nationality, age in years, sex, clinical antecedents, insulin use, and diabetes time. Our primary objective is to promote diversity in ophthalmology datasets and reduce under-represented countries in the pool used for model development.

## Dataset Description
The BRSET dataset is publicly available on PhysioNet, and you can access it through the following DOI link:

- **PhysioNet:** [A Brazilian Multilabel Ophthalmological Dataset (BRSET)](https://doi.org/10.13026/xcxw-8198)

Please refer to the PhysioNet page for detailed information on the dataset structure, contents, and citation guidelines.

## Usage
To use the BRSET dataset and perform technical validation, data analysis, quality assessment, and modeling, you can follow these steps:

1. Clone this repository to your local machine:
```
git clone https://github.com/luisnakayama/BRSET.git
```

2. Set up your Python environment and install the required libraries by running:

The Python version used here is `Python 3.8.17`
```
pip install -r requirements.txt
```

3. Explore the dataset and access the data for your analysis.

## Data Analysis
The data analysis for the BRSET can be found in the `eda.ipynb` notebook. It includes exploratory data analysis, plots, distributions, and an overview of the dataset. Feel free to use this notebook as a starting point for your own analysis.

## Quality Assessment
The quality assessment for the BRSET can be found in the `eda.ipynb` notebook. This section covers aspects such as missing values, data quality assessment, and the identification of duplicates. You can also generate a profiling report in the `Profile/brset_profiling.html` file for further insights into the data quality.

## Modeling
To build and evaluate models using the BRSET dataset, we provide a step-by-step guide in the `modeling.ipynb` and `modeling_embeddings.ipynb` notebook. These notebook demonstrates how to load the dataset, preprocess the images and labels, and train machine learning models for various tasks related to ophthalmology research.

### Pre-trained Embeddings
To facilitate the use and reduce computational costs, we offer pre-trained embeddings extracted from various backbone models. These embeddings can be extracted and used in ml tasks using the `modeling_embeddings.ipynb` file. You can find embeddings for the following backbone models:

- 'dinov2_small'
- 'dinov2_base'
- 'dinov2_large'
- 'dinov2_giant'
- 'clip_base'
- 'clip_large'
- 'convnextv2_tiny'
- 'convnextv2_base'
- 'convnextv2_large'
- 'convnext_tiny'
- 'convnext_small'
- 'convnext_base'
- 'convnext_large'
- 'swin_tiny'
- 'swin_small'
- 'swin_base'
- 'vit_base'
- 'vit_large'

These pre-trained embeddings can be utilized in your machine learning models to expedite the development and reduce computational overhead.


## Citation
If you use the BRSET dataset in your research, please cite the following publication:

**Physionet:** *Nakayama, L. F., Goncalves, M., Zago Ribeiro, L., Santos, H., Ferraz, D., Malerbi, F., Celi, L. A., & Regatieri, C. (2023). A Brazilian Multilabel Ophthalmological Dataset (BRSET) (version 1.0.0). PhysioNet. https://doi.org/10.13026/xcxw-8198.*

For more information about the BRSET dataset, please refer to the (PhysioNet link)[https://physionet.org/content/brazilian-ophthalmological/1.0.0/].