# Vehicle Recognition for Video Analysis and Summarization
## Introduction
Vehicle Make and Model Recognition (VMMR) has evolved into a signiﬁcant subject of study due to its importance in numerous Intelligent Transportation Systems (ITS), such as autonomous navigation, traffic analysis, traffic surveillance and security systems. A highly accurate VMMR system is of a great need, in order to optimize the costs and the human workload. The VMMR problem is a ﬁne-grained multi-class classification task with a peculiar set of issues.

This repository enables to train and evaluate a VMMR model that is able to predict the make, model and manufacture year of a vehicle based on its image. For this purpose, a fine-grained large-scale dataset with 2522 diverse classes was used. This repository also proposes a comprehensive system which can be used for automated vehicular surveillance. The VMMR system takes as input a surveillance video footage, analyzes it by first detecting vehicles and people (using [YOLO](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)), tracks the vehicles (using [SORT](https://github.com/abewley/sort)) throughout the video, performs the VMMR task to predict the make, model and manufacture year, recognizes the colour (by clustering the RGB triplets) and finally summarizes the video by generating a database containing all the extracted metadata. The system is made end-user through a Graphical User Interface that additionally allows the user to launch queries to look for a specific vehicle based on descriptions. This can be regarded as an efficient tool for law enforcement agencies that enables them to automate, improve and accelerate the search for suspicious vehicles (See demo [here](https://drive.google.com/file/d/1kJOdUlnQsFEwyUu6nr0yjLFTbqbTzeua/view?usp=sharing))

## License
This repository is released under the GNU License (refer to the LICENSE file for details) to promote the open use of the system and future improvements.

## Dependencies
This repository uses a Pytorch environment. To run it, you need to install:
- Torch with GPU (the used version is 1.3.1, any updated version should run fine)
- opencv-contrib-python (the used version is 4.5.1)

## How to run this code
### Vehicle classifier
#### Testing
To evaluate the pretrained VMMR model, first [download](https://drive.google.com/drive/folders/1kFRZNAPPry7AAlq9F5L2Nuf-AtZNcg6o?usp=sharing) the pretrained weights and place them under the directory "./VMMR_model". The pretrained model uses a ResNet50 CNN architecture. So, make sure to specify the right architecture name in the parameter "model" in "./config.json". Finally, run:

`$ python test_VMMR.py`
#### Training
To train the model from scratch, you need to download the vehicle dataset. Two datasets were used: [VMMRdb dataset](https://github.com/faezetta/VMMRdb) and [CompCars dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/). In this work, the two datasets were merged based on the common classes. Then the classes with less than 30 images were filtered out. Finally, the resulting dataset that was used in this project consists of 2522 classes. 

You need to have two folders under "./data". The first one must be entitled "train" which containes subfolders of the classes names. Each subfolder contains the training images for that class. The second folder must be entitled "val" which containes the data images that will be used to evaluate the trained model throughout the training in order to pick the best performing one.

Finally, in "./config.json", change the parameters related to the training as wished, and run:

`$ python train_VMMR.py`

The best performing model will be saved in "./VMMR_model/best_model.pth". Figures of the loss and accuracy evolution through the epochs are also saved under the same directory.
### GUI
See demo [here](https://drive.google.com/file/d/1kJOdUlnQsFEwyUu6nr0yjLFTbqbTzeua/view?usp=sharing).

To use the GUI, you need to [download](https://drive.google.com/drive/folders/1e2Vus6Gcx6PkvJEvnIiXMOmwla2aLSdF?usp=sharing) the pretrained YOLO weights and place them in the right directory "./utils/yolo-coco", then run:

`$ python run_GUI.py`

Any output saved by the GUI can be found in "./GUI_output" that will be created when the GUI is launched.
