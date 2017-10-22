import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers  import Conv2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils,plot_model

import inception_v4

from keras.datasets import mnist
from matplotlib import pyplot as plt
import drdata

def main():
    data = drdata.get_train_img()
    data.print_stat()

    preprocess(data)
    data.r_s_split(ratio= 0.2)

   # model = build_model(data)
    model = inception_v4.create_inception_v4((data.ch,data.w,data.h),nb_classes = 5)
    print "Fitting model with this structure:"
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(data.trn[0], data.trn[1], 
            batch_size=32, epochs=200, verbose=1)
    
 #   plt.plot([v['val_acc'] for v in model.history])
 #   plt.show()

    print "\nEvaluating model..."
    score = model.evaluate(data.tst[0], data.tst[1], verbose=1)
    count_acc_by_hand(50,model,data)
    print "loss:%f , score:%f "%(score[0],score[1])

def count_acc_by_hand(cnt,model,data):
    orig =  data._lb.inverse_transform(data.trn[1])[:cnt]
    print "\noriginal labels ",orig
    pr =  np.array([cls_of_pred( model.predict(np.array([data.trn[0][i]]))[0]) for i in range(cnt)])
    print "predicted labels", pr
    print "Example of prediction",model.predict(np.array([data.trn[0][0]]))
    print "accuracy by hand:",sum([pr[i]==orig[i]for i in range(cnt)])/(cnt*1.0)


def build_model(d):
    # building model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides = 3,activation='relu', input_shape=(d.ch,d.h,d.w),dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu',dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu',dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu',dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))
    model.add(Conv2D(64, (3, 3), activation='relu',dim_ordering='th'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(d.cls, activation='softmax'))
    return model

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
    d.X = d.X.reshape(d.len, d.ch, d.h, d.w).astype('float32')
    d.X /= 255

main()

