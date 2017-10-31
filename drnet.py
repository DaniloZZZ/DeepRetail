
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def drnet(input_shape,nb_classes):
	x  = Sequential()
	x.add(Conv2D(256, (7,7), strides = 3,activation ='relu',input_shape=input_shape))
	x.add(MaxPooling2D(pool_size=(5,5)))
	x.add(Dropout(0.25))
	x.add(Conv2D(128,(3,3),activation='relu'))
	x.add(Dropout(0.25))
	x.add(Conv2D(64,(3,3),activation='relu'))
	x.add(Conv2D(64,(3,3),activation='relu'))
	x.add(Conv2D(64,(3,3),activation='relu'))
	x.add(Conv2D(64,(3,3),activation='relu',name = "lconv"))

	x.add(Flatten())
	x.add(Dense(64,activation='relu'))
	x.add(Dropout(0.5))
	x.add(Dense(64,activation='relu'))
	x.add(Dropout(0.5))
	x.add(Dense(nb_classes,activation='softmax',name="preds"))
	return x

