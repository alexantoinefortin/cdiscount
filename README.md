### cdiscount Kaggle Image Classification Competition
(https://www.kaggle.com/c/cdiscount-image-classification-challenge)

The goal of the competition is to classify images of products (180x180pixels) into their respectives category. 

#### Specifics
9M products, 15M images, 5K categories

#### Details
Use `ubuntu-setup.sh` to setup your environment (optional)

Run `00-TrainTest-Extraction.py` to extracts images from train.bson

Run `01-Xception-Model.py` to train the top-layer of the Xception architecture used the published weights for ImageNet.

Run `01-Xception-ResumeTrain1.py` to resume training of several top-layers of the Xception model

Run `02-prediction-gpu.py` to get your predictions

#### Results
Running this code yields a model with 0.65663 accuracy. Due to a mistake early on during the project, I only extracted 1 image by `_id`. Quite often, an `_id` have more than one `_pics` associated with it. Somehow, I overlooked this and extracted only 1 image per `_id` for my training/testing set.

#### Things that was new for me during this project
* Number of images
* Number of categories
* Using and coding generators
* Loading and using the pre-trained Xception architecture
* Scoring using a 3-crops validation
* The `.bson` format

#### Dependencies
* Python 3.6.3
* tensorflow==1.4.1
* Keras==2.1.1
* h5py==2.7.1
* numpy==1.13.3
* tqdm==4.19.4


