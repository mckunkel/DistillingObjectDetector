# DistillingObjectDetector
Insight Project of Distilling an Neural Network, which is a method to transfer knowledge from a larger teacher model into a student model
For a indepth video lecture by Geoffer Hinton see
https://www.youtube.com/watch?v=EK61htlw8hY
based upon the idea of
https://arxiv.org/abs/1503.02531

## Requirements
### Anaconda Environment
* see [Setting a conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
* conda environment included as environment.yml
*  conda env create -f environment.yml
* conda activate Distilling


## Usage
### Data (folder data_stuff)
#### Get the Data
* python3 run_scripts.py
### Split the data
* python3 split_but_no_resizing.py
### Get a new set of weights with the CalTech256 image data set
* python3 train_xception.py
** This should also save several metric plots
*** top5_accuracy_vs_epoch.png
*** accuracy_vs_epoch.png
*** logloss_vs_epoch.png

