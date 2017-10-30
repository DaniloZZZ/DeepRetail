
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def drnet(input_shape,nb_classes):
	x  = Sequential()
	x.add(Conv2D(32, (5,5), strides = 1,activation ='relu',input_shape=input_shape))
	x.add(MaxPooling2D(pool_size=(2,2)))
	x.add(Conv2D(32,(3,3),activation='relu'))
	x.add(Dropout(0.25))
	x.add(Conv2D(32,(3,3),activation='relu'))
	x.add(MaxPooling2D(pool_size=(2,2)))
	x.add(Conv2D(32,(3,3),activation='relu'))
	x.add(MaxPooling2D(pool_size=(2,2)))
	x.add(Conv2D(32,(3,3),activation='relu'))
	x.add(MaxPooling2D(pool_size=(2,2)))
	x.add(Conv2D(32,(3,3),activation='relu',name = "lconv"))
	x.add(MaxPooling2D(pool_size=(2,2)))

	x.add(Flatten())
	x.add(Dense(16,activation='relu'))
	x.add(Dropout(0.5))
	x.add(Dense(nb_classes,activation='softmax',name="preds"))
	return x

