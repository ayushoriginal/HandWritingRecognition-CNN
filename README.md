# Digit Recognizer
This is an implementation of a convolutional neural network for recognizing hand written digits using the MNIST dataset. This model attains a validation accuracy of about 99.2% is obtained after training for 12 epochs. The model architecture and weights are saved in the files `model_architecture.json` and `model_weights.h5`. Note that these weights are compatible only with the Tensorflow backed.

To train the model run `train.py`. The file `test.py` generates a file `predictions.csv` which contains the predicted labels to the images in the test set. This file can be used for submission at Kaggle. `display_random.py` displays 25 random images from the test set along with their predicted labels. Here is an example:

<img src="https://github.com/Shobhit117/digit-recognizer/blob/master/figure_1.png" height=300px width=400px>

## Requirements

* Python 2.7
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [h5py](http://www.h5py.org/)
* numpy
* matplotlib
* pandas

## Dataset

* The model is trained on the MNIST dataset downloaded from [Kaggle](https://www.kaggle.com/c/digit-recognizer). 

* The file `train.csv` contains pixel intensity values as flattened vectors for 42000 images and their corresponding labels. Similarly, `test.csv` has pixel intensity values for 28000 unlabelled images.

## The Model

<img src="https://github.com/Shobhit117/digit-recognizer/blob/master/model.png">


