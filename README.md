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
git clone https://github.com/your-username/brset.git
```

2. Set up your Python environment and install the required libraries by running:

The Python version used here is `Python 3.8.17`
```
pip install -r requirements.txt
```

3. Explore the dataset and access the data for your analysis.

## Data Analysis
Use the BRSET dataset for data analysis to gain insights into the ophthalmological data. You can perform various exploratory data analysis (EDA) tasks, such as statistical summaries, data visualization, and understanding the distribution of various attributes.

## Quality Assessment
Conduct a quality assessment of the dataset to identify any anomalies, inconsistencies, or errors in the data. Quality assessment is crucial for ensuring the reliability and accuracy of the dataset before building models.

## Modeling
Leverage the BRSET dataset to develop and train machine learning or deep learning models for ophthalmological tasks. You can use the labeled retinal images and associated demographic information to train and evaluate your models.

## Citation
If you use the BRSET dataset in your research, please cite the following publication:

**Physionet:** *Nakayama, L. F., Goncalves, M., Zago Ribeiro, L., Santos, H., Ferraz, D., Malerbi, F., Celi, L. A., & Regatieri, C. (2023). A Brazilian Multilabel Ophthalmological Dataset (BRSET) (version 1.0.0). PhysioNet. https://doi.org/10.13026/xcxw-8198.*

For more information about the BRSET dataset, please refer to the (PhysioNet link)[https://physionet.org/content/brazilian-ophthalmological/1.0.0/].