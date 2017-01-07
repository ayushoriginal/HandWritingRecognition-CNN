import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from test import get_model,get_mnist_test_data
import time

X_test = get_mnist_test_data()
model = get_model()

# Get 25 random numbers between 0 and 28000
random_idx = np.random.random_integers(0,28000,size=25)

# Predict for the randomly selected 25 images
y_test = np.argmax(model.predict(X_test[random_idx],verbose=0),axis=1)

# A function to generate a grid of the 25 images with their corresponding predicted values
def generate_grid():
    plt.figure()
    for i in range(25):
        plt.subplot(5,5,i)
        plt.axis('off')
        plt.imshow(X_test[random_idx[i],0,:,:],cmap=cm.binary)
        plt.text(3,5,str(y_test[i]),fontsize=15,bbox={'alpha':0.3,'pad':9})
    plt.show()

# Display the images with their predicted values at intervals of 5 seconds
for i in range(25):
	plt.figure()
	plt.axis('off')
	plt.imshow(X_test[random_idx[i],0,:,:],cmap=cm.binary)
	plt.text(3,4,str(y_test[i]),fontsize=40,bbox={'alpha':0.5,'pad':10})
	plt.show(block=False)
	time.sleep(5)
	plt.close()

# Display the grid of images and their predicted values
generate_grid()
