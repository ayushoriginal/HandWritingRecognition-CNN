import numpy as np
import pandas as pd
from keras.models import model_from_json

def get_model():
    # Load the model architecture
    model = model_from_json(open('model_architecture.json').read())
    # Load the model weights
    model.load_weights('model_weights.h5')
    return model

def get_mnist_test_data():
    data = pd.read_csv('Data/test.csv')
    images = data.values
    images = np.multiply(images,1.0/255.0)
    images = images.reshape(images.shape[0],1,28,28)
    return images

if __name__=='__main__':
    X_test = get_mnist_test_data()
    model = get_model()
    y_test = np.argmax(model.predict(X_test,batch_size=50,verbose=1),axis=1)
    # Save the predictions:
    np.savetxt('predictions.csv',np.c_[range(1,X_test.shape[0]+1),y_test],delimiter=',',header = 'ImageId,Label',comments = '',fmt='%d')
    print('Predictions File Generated!')