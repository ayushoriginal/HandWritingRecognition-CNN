import numpy as np
import pandas as pd
import os

def flat_to_one_hot(labels):
    num_classes = np.unique(labels).shape[0]
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

def get_mnist_data(validation_size=2000):
	data = pd.read_csv('Data/train.csv')
	images = data.iloc[:,1:].values
	labels = data[[0]].values.ravel()
	# Convert the images from uint8 to double:
	images = np.multiply(images,1.0/255.0)
	# Convert the labels to one hot encoding:
	labels = flat_to_one_hot(labels)
	# Split the data into validation and training data:
	validation_images = images[:validation_size]
	validation_labels = labels[:validation_size]
	train_images = images[validation_size:]
	train_labels = labels[validation_size:]
	# Convert the images from flat to matrix form:
	train_images = train_images.reshape(train_images.shape[0],1,28,28)
	validation_images = validation_images.reshape(validation_images.shape[0],1,28,28)
	# Return the data:
	return (train_images,train_labels),(validation_images,validation_labels)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

if __name__ == '__main__':
    (X_train,y_train),(X_val,y_val) = get_mnist_data()
    # Construct the model:
    model = Sequential()
    model.add(Convolution2D(nb_filter=32,nb_row=5,nb_col=5,border_mode='same',input_shape=(1,28,28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
    model.add(Convolution2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same',input_shape=(32,14,14)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    if os.path.exists('./model_weights.h5'):
        model.load_weights('model_weights.h5')
    # Train the model:
    model.fit(X_train,y_train,batch_size=50,nb_epoch=1,verbose=1,validation_data=(X_val,y_val))
    score = model.evaluate(X_val,y_val,verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])
    # Save the model:
    json_string = model.to_json()
    open('model_architecture.json','w').write(json_string)
    # Save the weights
    model.save_weights('model_weights.h5',overwrite=True)


