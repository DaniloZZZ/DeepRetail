import io,os
import numpy as np
from keras.models import Sequential,model_from_json
from keras.preprocessing.image import ImageDataGenerator
import vis.visualization as visualization
from vis.utils import utils as vutils
from sklearn.preprocessing import Normalizer

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
		prtnt "Creating model \"%s\""%name
		self.model_name = name

		d = drdata.get_train_img()
		self.data = d
		self.__model_dir = self.__models_dir+name +"/"

		print "Normalizing data, each sample has L2 of one"
		self._normer = Normalizer()
		self._normer.fit(d.X)
		self._normer.transform(d.X)

# -- Training --
	def train_augm(self,epochs=20):
		self.train_datagen = ImageDataGenerator(
				rotation_range=70,
				width_shift_range= 0.15,
				height_shift_range = 0.15,
				shear_range=0.2, zoom_range=0.2,
				horizontal_flip=True)

		train_gen = self.train_datagen.flow_from_directory(
				self.__train_dir,
			#	save_to_dir = self.__augmented_dir,
				target_size=self.im_size,
				batch_size=16,
				class_mode='categorical')
		validation_generator= self.train_datagen.flow_from_directory(
				"data/validation",
			#	save_to_dir = self.__augmented_dir,
				target_size=self.im_size,
				batch_size=16,
				class_mode='categorical')

		self.data.r_s_split(ratio = 0.33)
		train_gen= self.train_datagen.flow(
				self.data.trn[0],
				self.data.trn[1],
				batch_size = 16)
		validation_generator = self.train_datagen.flow(
				self.data.tst[0],
				self.data.tst[1],
				batch_size = 16)
		for x,y in train_gen:
			plt.imshow(x[0])
			print self.data.get_label_names(y)[0]
			plt.show()
			break
		for x,y in validation_generator:
			plt.imshow(x[0])
			print self.data.get_label_names(y)[0]
			plt.show()
			break

		self.model.fit_generator(
			train_gen,
			workers=1,
			steps_per_epoch=2000,
			epochs=epochs,
			validation_data=validation_generator,
			validation_steps=30)

	def train_classic(self,epochs = 20,subset=0):
		if subset<1: 
			subset= len(self.data.trn[1])
		print self.data.trn[1][1]
		plt.imshow(self.data.trn[0][1])
		plt.show()
		self.model.fit(self.data.trn[0][:subset],
			self.data.trn[1][:subset],
			epochs=epochs)

	def predict(self,img_arr):
		# takes binary data just from POST request		
		# load from binary to Image obj
		images = [Image.open(io.BytesIO(bts)) for bts in img_arr]
		# convert to np array and resize
		imgs = [cv2.resize(np.array(img),self.im_size) for img in images]

		#reshaping channels
		imgs = np.array(imgs)
		s = imgs.shape
		imgs = np.array(imgs).reshape(s[0],s[1],s[2],s[3])

		# Appplying Normalizer
		imgs = self._normer.transform(imgs)

		print "Predicting for %d images"%len(imgs)
		preds = self.model.predict(np.array(imgs))
		# get labels from prefictions
		labels = self.data._lb.inverse_transform(preds)
		return self.data.get_label_names(labels)

# -- Evaluating and visualizing model
	def save_activation_map(self,name):
		d = self.__model_dir+"activations/"
		check_create_dir(d)	
		# check data format and find layer by name
		if (type(name)==type("sds")):
			l_idx =vutils.find_layer_idx(self.model,
							name)
		else:
			l_idx = name

		print "layer idx",l_idx
		layer = self.model.layers[l_idx]
		out_shape = layer.output_shape
		name = layer.name
		print "output shape:",out_shape
		f_ids = range(out_shape[-1])[:10]
		for fi in f_ids:
			im = visualization.visualize_activation(
				self.model,
				l_idx, fi,
			input_range=(0.0,1.0))
			print fi,im.shape
			cv2.imwrite(d+"act_%s_F%d.png"%(name,fi),im*255)

	def save_saliency_map(self,name):
		d = self.__model_dir+"saliency/"
		# check data format and find layer by name
		if (type(name)==type("sds")):
			l_idx =vutils.find_layer_idx(self.model,
							name)
		else:
			l_idx = name

		print "layer idx",l_idx
		layer = self.model.layers[l_idx]
		out_shape = layer.output_shape
		name = layer.name
		print "output shape:",out_shape
		f_ids = range(out_shape[-1])[:10]
		for fi in f_ids:
			im = visualization.visualize_saliency(
				self.model,
				l_idx, fi,
				self.data.X[0])
			print fi,im.shape
			cv2.imwrite(d+"sail_%s_F%d.png"%(name,fi),im*255)


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


