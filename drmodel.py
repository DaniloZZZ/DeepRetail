import io,os
import numpy as np
from keras.models import Sequential,model_from_json
import cv2
import Image

import drnet, drdata

class DrModel:
	__models_dir = "models/"
	im_size = (299,299)
	def __init__(self):
		d = drdata.get_train_img()
		self.model =drnet.drnet((d.ch,d.w,d.h),5)
		self.data = d

	def predict(self,img_arr):
		# takes binary data just from POST request		
		# load from binary to Image obj
		images = [Image.open(io.BytesIO(bts)) for bts in img_arr]
		# convert to np array and resize
		imgs = [cv2.resize(np.array(img),self.im_size) for img in images]

		print "Predicting for %d images"%len(imgs)

		imgs = np.array(imgs)
		s = imgs.shape
		imgs = np.array(imgs).reshape(s[0],s[3],s[1],s[2])
		preds = self.model.predict(np.array(imgs))
		# get labels from prefictions
		labels = self.data._lb.inverse_transform(preds)
		return self.data.get_label_names(labels)

	def load_model(self,name):
		d = self.__models_dir+name
		check_create_dir(d)
		jsn = open(d+"/model.json")
		model_jsn = jsn.read()
		jsn.close()
		self.model = model_from_json(model_jsn)	
		self.model.load_weights(d+"/weights.h5")
		self.model_name= name
		print "Loaded Model from %s"%d
		
	def save_model(self):
		d = self.__models_dir+self.model_name
		check_create_dir(d)
		model_jsn = self.model.to_json()
		with open(d+"/model.json",'w+') as jsn:
			jsn.write(model_jsn)
		self.model.save_weights(d+"/weights.h5")	
		print "Saved model to %s"%d

def check_create_dir(d):	
    try:
        os.stat(d)
    except:
        os.mkdir(d,0o777)
        print "Created new directory %s "%(d)


