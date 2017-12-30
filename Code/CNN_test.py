
import gensim, logging
import glob


p#ath1 = '/home/nikil/'

X_test = []
len_test = []

for i in range(5):
    path = path1 + 'test' + str(i) + '/*.txt'
    files=glob.glob(path)
    a = []
    count=0
    for file in files:
        f=open(file,'r')
        cont = f.readlines()
        a.extend(cont)
        f.close()
    b=filter(lambda x: x != '\n', a)
    c = [s.rstrip() for s in b]
    count = len(c)
    X_test.extend([s.split(" ") for s in c])
    len_test.append(count)



# Preparing the test labels

#y_test = []
#for i in range(5):
#    y_test.extend([i] * len_test[i])

y_test = []
for i in range(5):
    temp = [0]*5
    temp[i]=1
    for i in range(len_test[i]):
        y_test.append(temp)



# Preparing the test data in the form of indices from model.vocab

X_test1 = []
for i in range(len(X_test)):
    temp = []
    s = X_test[i]
    for word in s:
        if word in model.vocab.keys():
            temp.append(model.vocab[word].index)
    X_test1.append(temp)

print(len(X_test1))



#Imports

import numpy as np
import theano
from numpy import array
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Activation, Lambda
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
#from keras.layers import LSTM, Input
from keras.models import Model
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import ModelCheckpoint


#import matplotlib
#import matplotlib.pyplot as plt
np.random.seed(1337)

# set parameters:

batch_size = 32
num_filter = 200
nb_epoch = 4
pool_length = 2
output_dim = 1
hidden_dims = 250
hidden_dims2 = 100
maxlen = 210
max_features = 5000
embb_size = dim


theano.config.floatX = 'float32'


X_test2 = sequence.pad_sequences(X_test1, maxlen=maxlen,padding = 'post')

X_test2 = X_test2.astype("float32")


model1 = Sequential()


model1.add(Embedding(len(model.vocab), embb_size, input_length=maxlen, weights = [array(embb_weights)]))


model1.add(Convolution1D(input_dim=embb_size, nb_filter = num_filter, filter_length = 4,
                        border_mode="valid", activation="relu",subsample_length=1))


model1.add(Convolution1D(input_dim=num_filter, nb_filter = 100, filter_length = 3 ,
                        border_mode="valid", activation="relu",subsample_length=1))


def max_1d(X):
    return K.max(X, axis=1)


model1.add(Lambda(max_1d, output_shape=(100,)))


model1.add(Dense(hidden_dims))
model1.add(Dropout(0.2))
model1.add(Activation('relu'))


model1.add(Dense(5))
model1.add(Activation('sigmoid'))


model1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights(‘/home/nikil/weights/checkpoint-02-0.46.hdf5') 

pred = model1.predict_classes(X_test2, verbose=0)


y_test1 = []
for i in range(5):
    y_test1.extend([i] * len_test[i])


count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count=0

for i in range(len(pred)):
     if(y_test1[i] == pred[i]):
        count = count + 1

print("\n" + str(count/float(len(pred))))

        
for i in range(len(pred)):
    
    if(y_test1[i] == 0 and pred[i] == 0):
        count0 = count0 + 1
        
    if(y_test1[i] == 1 & pred[i] == 1):
        count1 = count1 + 1
        
    if(y_test1[i] == 2 & pred[i] == 2):
        count2 = count2 + 1

    if(y_test1[i] == 3 & pred[i] == 3):
        count3 = count3 + 1

    if(y_test1[i] == 4 & pred[i] == 4): 
        count4 = count4 + 1


print("\n" + str(count0/float(len_test[0])))
print(count1/float(len_test[1]))
print(count2/float(len_test[2]))
print(count3/float(len_test[3]))
print(count4/float(len_test[4]))

