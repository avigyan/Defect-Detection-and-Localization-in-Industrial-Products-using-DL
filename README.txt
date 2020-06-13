AUTHOR: AVIGYAN SINHA
USAGE: python3 defect_detect.py --image bad_image.png --model defect_detect.model

DESCRIPTION: This project detects any type of metallic defects present in an image. The folder contains the following:

1) data_aug.py - takes a an input imgae and generates new images for training using data-augmentation. Input image should have directory structure as "dataset/class/image.png"

2) fine_tune_train.py - trains(fine-tunes by transfet learning, a modified VGG16 architecture pretrained on ImageNet dataset). Removes all FC layers in the VGG16, adds a Global Average Pool(GAP) layer followed by sigmoid/softmax layer. Saves the trained model to disk

3) fcheadnet.py - defines new head layers added to the VGG16 after removing its FC layers

4) evaluate.py - takes test images and evaluates the trained model

5) defect_detect.py - takes the trained model and a new test image. Draws a bounding box around defect if present in the image. Prints whether defect present on not on Linux terminal

6) inspect_model.py - utulity function to display the layers present in a model

7) requirements.txt - list the Python 3.6.9 libraries required for this project

The folder also contains some test images

NOTE: To train on new defect classes simply input them to data_aug.py and generate training images for each class. Than train on them using fine_tune_train.py. If test images contain defects at new relative positions, then include images having same defect class with similar relative positions in training data also. 
