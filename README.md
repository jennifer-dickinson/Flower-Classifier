# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Author

Jennifer Salas <br>
November 2019

## Submission Files

### Part 1

- Image_Classifer_Project.html

### Part 2

- train.py
- predict.py
- helper.py
- model.py

## Train.py

Train a neural network

    usage: train.py [-h] [--save_dir SAVE_DIR] [--learning_rate LEARNING_RATE]
                    [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS] [--arch ARCH]
                    [--gpu]
                    data_directory
                    
    positional arguments:
      data_directory

    optional arguments:
      -h, --help            show this help message and exit
      --save_dir SAVE_DIR
      --learning_rate LEARNING_RATE
      --hidden_units HIDDEN_UNITS
      --epochs EPOCHS
      --arch ARCH
      --gpu
      
 ## Predict.py
 
 Predict an image from a pretrained neural network
 
     usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                      [--gpu]
                      image_path checkpoint

    positional arguments:
      image_path
      checkpoint

    optional arguments:
      -h, --help            show this help message and exit
      --top_k TOP_K
      --category_names CATEGORY_NAMES
      --gpu
