import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import data from file
train_data = pd.read_csv("train.csv")

labels = train_data.iloc[:,[0]].values
images = train_data.iloc[:,1:].values

images = images/255.0 #to normalise each image to [0,1]


#images=images.values.reshape(-1,28,28,1)
from keras.utils import to_categorical
labels = to_categorical(labels,num_classes = 10) #to convert labels as 3= [0,0,0, 1, 0, 0, 0,0 ,... ]

train_im, test_im, train_labels,test_labels = train_test_split(images, labels, test_size = 0.1, random_state=2)
train_im = train_im.reshape(train_im.shape[0], 28, 28, 1).astype('float32')
test_im = test_im.reshape(test_im.shape[0], 28, 28, 1).astype('float32')
plt.imshow(test_im[0][:,:,0])
plt.show()

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D  #to add convolutional layer
from keras.layers.convolutional import MaxPooling2D #to add pooling layer

def convolutional_model(num_classes):
    model = Sequential()
    input_shape = (28,28,1)
    
    model.add(Conv2D(32,kernel_size=(2,2),strides=(1,1),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(64,kernel_size=(2,2),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    
    #compile and optimize the model 
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    
    
num_classes = 10 
model = convolutional_model(num_classes)

#fitting data images and labels to CNN model
model.fit(train_im,train_labels,batch_size=100,epochs=10,verbose=2)

#evaluating model
scores = model.evaluate(test_im, test_labels, verbose=0)

print('Accuracy :',scores)
