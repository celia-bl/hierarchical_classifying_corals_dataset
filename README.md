# Hierarchical Classification for Automated Image Annotation of Coral Reef Benthic Structures

## Overview
This repository contains the implementation and results of a comparative study between hierarchical and flat classifiers using a new dataset of Brazilian benthic images in Rio do Fogo, and also two well-known benthic datasets MLC, and TasCPC. The aim is to evaluate the performance of these classifiers and determine which approach yields better results for the given problem. It conducted to a poster presentation of an article titled "Hierarchical Classification for Automated Image Annotation of Coral Reef Benthic Structures" in the workshop CCAI at NeurIPS conference (link will be available after the conference). 

## Datasets 
### Rio do Fogo 
This custom dataset comprises 1,549 images collected and annotated by the Marine Ecology Laboratory/UFRN at Rio do Fogo, Northeast Brazil. Between 2017 and 2022, 100 images were collected from each of two sites every three months. Each image contains 25 random annotated points, resulting in 38,725 annotations across 54 distinct labels, which are highly imbalanced. Specifically, 11 labels account for 95% of the annotations, while 24 classes have fewer than 20 annotations each. 
The images are available upon request at celia.blondin@ird.fr

### Moorea Labeled Corals (MLC) 
The Moorea Labeled Corals dataset is a subset of the Moorea Coral Reef-Long Term Ecological Research (MCR-LTER) dataset packaged for Computer Vision research. It contains over 400.000 human expert annotations on 2055 coral reef survey images from the island of Moorea in French Polynesia. Each image has 200 expert random point annotations, indicating the substrate underneath each point. Introduced by [Beijbom et al in CVPR 2012](https://www.researchgate.net/publication/261494296_Automated_Annotation_of_Coral_Reef_Survey_Images). 

[The website where you can download this dataset.](https://portal.edirepository.org/nis/mapbrowse?packageid=knb-lter-mcr.5006.3) 

### TasCPC 
The data set is comprised of 22 dive missions conducted by AUV Sirius off the South-East coast of Tasmania in October 2008. It contains over 60,000 annotations on 1258 images. Each image contains 50 annotations, indicating biological species (including types of sponge, coral, algae and others), abiotic elements (types of sand, gravel, rock, shells etc.), and types of unknown data (ambiguous species, poor image quality, etc.).

[The website where you can download this dataset.](http://marine.acfr.usyd.edu.au/datasets/) 


## Hierarchy 
### Rio do Fogo 
![hierarchy_graph](https://github.com/user-attachments/assets/42d34094-1edd-4570-8d2b-e2068f1da4f6)

### MLC

### TasCPC

## Feature Extraction
To extract features, a 224x224 image patch is cropped around each annotated point. This process is handled in `img_transformation.py`. These patches are then processed through an EfficientNet B0 model, with weights customized by CoralNet, trained on 16 million benthic images. The resulting feature vectors are used for training the classifiers.
The weights can be download [here](https://spacer-tools.s3.us-west-2.amazonaws.com/efficientnet_b0_ver1.pt), provided by [CoralNet](https://coralnet.ucsd.edu/). 

## Flat Classifier
Our baseline is a standard Multi-Layer Perceptron (MLP) classifier provided by the scikit-learn library ([scikit-learn](https://github.com/scikit-learn/scikit-learn.git)). The MLP has two neural network layers with 200 and 100 units, respectively.

## Hierarchical Classifier
The hierarchical classifier is implemented using `hiclass` ([hiclass](https://github.com/scikit-learn-contrib/hiclass.git)), which also depends on scikit-learn classifiers. We opted for a Local Classifier per Parent Node, meaning a separate classifier for each node with children. Each of these classifiers is an MLP identical to our baseline, with two layers of 200 and 100 units.

## Metrics
The metrics used to compare the two approaches are the F1-score and a hierarchical score. The F1-score is computed using scikit-learn, while the hierarchical score is computed using `hiclass`. The hierarchical score penalizes errors less when elements are close in the hierarchical tree. You can read more about the metrics [here](https://hiclass.readthedocs.io/en/latest/algorithms/metrics.html).

## Setup 
To install all the package dependency : 
```bash
$ conda create --name <env> --file <this file>
```
Then configure the configuration file : 
### params.yaml
```yaml
# Dataset configuration
# Available options: RIO, MLC, TasCPC
dataset: "RIO"

# Crop size for patches, strictly limited to 224 for efficientnet-b0
crop_size: 224

# Path where the images are stored
image_path: "./Images_RIO"

# Path where you want to store the extracted features
features_path: './.features'

# Path to annotations file (required for RIO and TasCPC datasets)
annotations_file: "./annotations/annotations.csv"

# Path to store the results
result_folder: './results'

# Name for the model used for feature extraction (efficientnet-b0 pretrained by CoralNet)
model_extraction_name: 'efficientnet-b0'

# Path to the pretrained weights file for efficientnet-b0 (by CoralNet)
weights_file: './weights/efficientnet_b0_ver1.pt'

# Labelset file (useful only for CoralNet datasets like RIO)
labelset_file: './labelset/labelset.csv'

# List of the number of training patches to use for training
list_nb_training_patches: [2, 20]

# Classifiers to use for training (currently supports MLP)
classifiers:
  flat_classifiers:
    - type: MLP
      params:
        hidden_layer_sizes: [200, 100]  # Sizes of the hidden layers for MLP

  hierarchical_classifiers:
    - type: LCPPN
      params:
        hidden_layer_sizes: [200, 100]  # Sizes of the hidden layers for LCPPN
```
1. Choose the dataset
2. Enter the right image path
3. Enter the path to the annotation file (for RIO or TASCPC)
4. Enter the path to the weight file
5. Enter the path to the labelset (if RIO)
6. Choose the number of training patches you want to train the classifiers (list_nb_training_patches)
### Extract and train 
To extract you can just : 
```bash
$ python3 main.py extract
```
To train : 
```bash
$ python3 main.py train
```
