
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from keras.utils.data_utils import get_file
def alexNet(input_shape,nb_classes):
	x  = Sequential()
	x.add(Conv2D(96, (11,11), strides = 4,activation ='relu',input_shape=input_shape))
	x.add(BatchNormalization())
	x.add(MaxPooling2D(pool_size=(2,2)))
	x.add(Conv2D(256, (5,5),activation ='relu'))
	x.add(BatchNormalization())
	x.add(MaxPooling2D( pool_size=(2,2)))
	x.add(Conv2D(384 ,(3,3),activation='relu'))
	x.add(Conv2D(384 ,(3,3),activation='relu'))
	x.add(Conv2D(256,(3,3),activation='relu'))
	
	x.add(MaxPooling2D(pool_size=(2,2)))
	x.add(Flatten())
	x.add(Dense(1024,activation='relu'))
	x.add(Dropout(0.5))
	x.add(Dense(1024,activation='relu'))
	x.add(Dropout(0.5))
	x.add(Dense(nb_classes,activation='softmax'))
	return x

