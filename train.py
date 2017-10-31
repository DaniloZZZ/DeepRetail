import os
import sys
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DEVICE"]="cuda0"

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers  import Conv2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils,plot_model
from keras import optimizers

import inception_v4, alexnet,drnet

from keras.datasets import mnist
from matplotlib import pyplot as plt
import drdata
from drmodel import DrModel 

def train():
    drm = DrModel()
    drm.data.r_s_split()
    data = drm.data

    drm.model_name = "drmodel"
    model =drnet.drnet((data.w,data.h,data.ch),data.cls)
    #model = inception_v4.create_inception_v4((data.ch,data.w,data.h),nb_classes = 5)
    #model = alexnet.alexNet((data.ch,data.w,data.h),5)
    print "Fitting model with this structure:"
    model.summary()
    
    print "Compiling model..."
    sgd  = optimizers.SGD(lr=0.02,decay = 1e-6,momentum=0.2, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

    drm.model = model
    drm.train_augm(epochs = 40)

  #  drm.train_classic(epochs=30,subset=100)
    orig =  data._lb.inverse_transform(data.trn[1])

    print "\nEvaluating model..."
    data.scale(1/255.)
    score = drm.model.evaluate(data.tst[0], data.tst[1], verbose=1)
    count_acc_by_hand(100,drm.model,data)
    print "loss:%f , score:%f "%(score[0],score[1])
    print  "saving model..."
    drm.save_model()

def train_minibatches():
    num_batches = 20
    bsize = 20
    learning_curve = []
    for e  in range(220):
	    print "epoch #%d"%e
	    loss = 0
	    acc = 0
	    for i in range(num_batches):
		    b_x = data.trn[0][i*bsize:(i+1)*bsize]
		    b_y =  data.trn[1][i*bsize:(i+1)*bsize]
		    print "batch #%d"%i
		    sys.stdout.write("\033[F")
		    model.train_on_batch(b_x, b_y)
			    
		    _loss, _acc = model.test_on_batch(b_x,b_y)
		    loss = loss+_loss
		    acc= acc+_acc
	    accuracy = acc/num_batches
	    learning_curve.append(accuracy)
   	    print "Accuracy: %f, loss: %f "%(acc/num_batches,loss/num_batches)

def count_acc_by_hand(cnt,model,data):
    orig =  data._lb.inverse_transform(data.tst[1])[:cnt]
    print "\noriginal labels ",orig
    pr =  np.array([cls_of_pred( model.predict(np.array([data.tst[0][i]]))[0]) for i in range(cnt)])
    print "predicted labels", pr
    print "Example of prediction",model.predict(np.array([data.tst[0][0]]))
    print "accuracy by hand:",sum([pr[i]==orig[i]for i in range(cnt)])/(cnt*1.0)

def cls_of_pred(prediction):
    c = 0
    i = 0
    ma = 0
    for p  in prediction:
        if p>ma:
            ma = p
            c = i
        i=i+1
    return c

def preprocess(d):
    # transform data to have width, normalize
    d.X = d.X.reshape(d.len, d.h, d.w, d.ch).astype('float32')
    d.X /= 255

train()

