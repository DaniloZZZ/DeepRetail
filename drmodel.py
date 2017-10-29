import io,os
import numpy as np
from keras.models import Sequential,model_from_json
from keras.preprocessing.image import ImageDataGenerator
import vis.visualization as visualization
from vis.utils import utils as vutils

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import drnet, drdata

class DrModel:
	__models_dir = "models/"
	__train_dir = "data/train/"
	__augmented_dir = "data/augm"
	im_size = (299,299)
	def __init__(self, name = "drmodel"):
		d = drdata.get_train_img()
		self.model =drnet.drnet((d.ch,d.w,d.h),5)
		self.data = d
		self.model_name = name
		self.__model_dir = self.__models_dir+name +"/"

	def train_augm(self):
		self.train_datagen = ImageDataGenerator(
				rescale = 1/255.,
				rotation_range=40,
				width_shift_range= 0.15,
				height_shift_range = 0.15,
				shear_range=0.2,
				zoom_range=0.2,
				horizontal_flip=True)

		train_gen = self.train_datagen.flow_from_directory(
				self.__train_dir,
			#	save_to_dir = self.__augmented_dir,
				target_size=self.im_size,
				batch_size=32,
				class_mode='categorical')

		self.data.r_s_split(ratio = 0.33)
		validation_generator = self.train_datagen.flow(
				self.data.tst[0],
				self.data.tst[1],
				batch_size = 32)

		self.model.fit_generator(
			train_gen,
			workers=4,
			steps_per_epoch=100,
			epochs=10,
			validation_data=validation_generator,
			validation_steps=800)

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
		preds = self.model.preict(np.array(imgs))
		# get labels from prefictions
		labels = self.data._lb.inverse_transform(preds)
		return self.data.get_label_names(labels)

# -- Evaluating and visualizing model
	def save_activation_map(self):
		d = self.__model_dir+"activations/"
		check_create_dir(d)	
		l_idx =vutils.find_layer_idx(self.model,"preds")
		print "preds idx",l_idx
		f_ids = [1,2,3]
		imgs = visualization.visualize_activation(
			self.model,
			l_idx, f_ids,
			verbose=True)
		n =0
		for im in imgs:	
			fi = f_ids[n]
			print fi 
			plt.imshow(im)
			plt.show()
			cv2.imwrite(d+"act_L%d_F%d.png"%(l_idx,fi),im)
			n=n+1

	def save_saliency_map(self):
		d = self.__model_dir+"saliency/"
		l_idx =5
		f_idx = [1]
		print "v"
		imgs = visualization.visualize_saliency(
			self.model,
			l_idx, f_idx,
			self.data.X[0])
		check_create_dir(d)	
		cv2.imwrite(d+"act_L%d_F%d.png"%(l_idx,f_idx[0]),imgs[0])


# -- Loading and saving model --

	# Loads model from __models_dir directory to self.model
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
		
	# Saves model from self.model to __models_dir directory 
	def save_model(self):
		d = self.__model_dir
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


