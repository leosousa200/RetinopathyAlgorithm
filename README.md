# RetinopathyAlgorithm

Diabetic vs Hypertensive Retinopathy Detection using Active Learning &amp; Transfer Learning

Project developed as part of the Artificial Intelligence course lectured at the University of Coimbra in the academic year of 2024/2025

### Database usage

Diabetic Retinopathy 224x224 Gaussian Filtered (dataset1): https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data

Ocular Disease Recognition (dataset2): https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data?select=full_df.csv

### Code organization

- trainFull: Train a CNN using the entire dataset1.
- trainAL: First train a model of CNN with an initial set of random seeds from dataset1, then perform Active Learning loop by iteratively adding new samples (those with the highest uncertainties).
- DR_evaluation: Compare the performance and data usage of the full model and the Active Learning (AL) model mentioned above.
- transferLearning_downsample: Perform Transfer Learning on the trainAL model (freezing the top 10 layers) to enable Hypertensive Retinopathy detection. This is done using dataset2, downsampling it to balance the database, followed by an evaluation of the results.
- dataAugmentation: Generate new images for dataset2 to address its imbalance.
- transferLearning_augmentated: Perform Transfer Learning on the trainAL model (also freezing the top 10 layers) to enable Hypertensive Retinopathy detection, using dataset2 plus images obtained from dataAugmentation, followed by an evaluation of the results.
- TL_evaluation: Compare the performance of both transfer learning models (downsampling vs. data augmentation).

### How to execute

#### Create python environment

##### In macOS

- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt

##### In Windows

- py -m venv .venv
- .\.venv\Scripts\activate

#### Install python libraries

- pip install -r requirements.txt
