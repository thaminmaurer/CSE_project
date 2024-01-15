# CSE - Semester Project - SAM vs. U-net: A Comparative Analysis for Advanced Road Segmentation
You can read the report [here](https://github.com/thaminmaurer/CSE_project/blob/main/CSE_Final_report.pdf).


This project focuses on the critical task of road segmentation, aiming to enhancethe effectiveness of vehicle detection in urban environments. The primary motivation behind this shift lies in the recognition that precise road segmentation serves as a foundational element for robust vehicle detection, enabling better navigation through complex urban landscapes.
To achieve this, we employed two cutting-edge deep learning models, SAM (Segment Anything Model) and U-net architecture.


## Data 
The datasets are available at the following links :
- [EPFL dataset](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files )
- [UAVid dataset](https://uavid.nl/) : run the draw.py file to generate the in depth analysis of the dataset and save all the bounding boxes of occluded vehicles.
- [UAVDT dataset](https://paperswithcode.com/dataset/uavdt)

## File organization  
In order to run the code you will need the following file organization :

    ├── Datasets
    │   ├── EPFL
    │   ├── UAV-benchmark-M
        │   └── draw.py
    │   └── uavid_v1.5_official_release_image
    ├── SAM
    │   └── Fine_tune_SAM.ipynb
    └── UNet 
        ├── training  
        │   ├── groundtruth    
        │   └── images     
        ├── check_points  
        ├── models  
        ├── logs  
        └── actual code files (.ipynb, .py)


Where   
- the folders in training are empty (they will be populated with the augmented training set). 


## Code 
### Unet 
The following files are needed/useful for understanding the UNet part, training the model and producing the results :  
- models.py, definitions of the different models used.  
- helpers.py, helper methods used in the other files.  
- DataAugmentation.ipynb, file used for augmenting the data set in a customizable manner.  
- Unet.ipynb, file used to train the model and produce the results.

#### Running the code  
In order to train a model, please start by installing the required libraries.
- Tensorflow
- Keras
- Numpy
- Math
- Random
- Tqdm
- Sklearn
- Matplotlib
- Skimage
- cudatoolkit version 11.2
- cudnn version 8.1.0  

Once this is done the following workflow can be used :   
1) Augment the data by running cells in DataAugmentation.ipynb until satisfied with the train set  
2) Tweak the constants at the start of Unet.ipynb (optional)
3) Run the rest of Unet.ipynb  
The best model can be downloaded [here](https://drive.google.com/file/d/1OU9e1pOKxl4dUlTnMYkhio36ZouLw6ed/view?usp=sharing) 

### SAM 
For a better understanding of the SAM part, you can download the original directory [here](https://github.com/facebookresearch/segment-anything)

Fine_tune_SAM.ipynb is used to fine tune the SAM model with the chosen dataset and produce the results.

#### Running the code  
In order to train a model, please start by installing the required libraries.
- Tensorflow
- Keras
- Numpy
- Math
- Random
- Tqdm
- Sklearn
- Matplotlib
- Skimage
- cudatoolkit version 11.2
- cudnn version 8.1.0  

Once this is done run the entire notebook and see the results. 

The best model can be downloaded [here](https://drive.google.com/file/d/1uBRaSFMrzQArnmEO8YwYAfCBvsWTbZ4E/view?usp=sharing) 


 ## Bibliography
1. facebookresearch "segment-anything" GitHub Repository, 2023. [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

2. Mateo762 "U-Net Road Segmentation: A Deep Learning Approach for Road Segmentation" GitHub Repository, 2023. [https://github.com/mateo762/unet-road-segmentation](https://github.com/mateo762/unet-road-segmentation)

3. bnsreenu "331_fine_tune_SAM_mito" GitHub Repository, 2023. [https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb](https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb)

