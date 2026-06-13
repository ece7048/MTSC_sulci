"""Keras builders for 3D classification backbones."""

from __future__ import division, print_function
import os

import tensorflow as tf
from resnet3d import Resnet3DBuilder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class create_3Dnet:
	"""Factory for Keras 3D CNN, attention, and ResNet classifier models."""

	def __init__(self, model,height,width,depth,channels,classes,name="",do=0.3, path="/home/mm2703/code/s3D/test/",backbone="simple_3d",paral='off',b_w='simple_3d_gpu_L_skeleton_3d_image_classification.h5'):
		self.model=model
		self.height=height
		self.width=width
		self.depth=depth
		self.channels=channels
		self.classes=classes
		self.path=path
		self.name=name
		self.do=do
		self.backbone=backbone                
		self.backb_w=b_w
		self.par=paral
	def model_builder(self):
		"""Build the configured Keras classifier model."""
		if self.model=="simple_3d":
			init_model=self.simple_3d()
			model=self.MLP(init_model)	
		elif self.model=='simple_MHL':
			init_model=self.tune_MHL(backbone=self.backbone,name=self.name,store_model=self.path,parallel=self.par)

			model_file=str(self.path + "/"+self.backb_w)
			if os.path.exists(model_file):
				print(model_file)
				init_model.load_weights(model_file,by_name=True, skip_mismatch=True)
			model=self.MLP(init_model)
		elif self.model=='3D_resnet_50':
			model = Resnet3DBuilder.build_resnet_50((self.height, self.width, self.depth, self.channels),self.classes )
		elif self.model=='double_3d':
			init_model=self.double_3d()
			model=self.MLP(init_model)
		else:
			print("no model is given")
		return model


	def simple_3d(self,backbone_use='off'):
		"""Build a 3D convolutional neural network model."""

		inputs = keras.Input((self.width, self.height, self.depth, 1))

		x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
		x = layers.MaxPool3D(pool_size=2)(x)

		x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
		x = layers.MaxPool3D(pool_size=2)(x)

		x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
		x = layers.MaxPool3D(pool_size=2)(x)

		x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
		x = layers.MaxPool3D(pool_size=2)(x)

		x = layers.GlobalAveragePooling3D()(x)
		x = layers.Dense(units=512, activation="relu")(x)
		x = layers.Dropout(self.do)(x)
		if backbone_use=='off':
			outputs = layers.Dense(units=1024, activation="softmax")(x)
		else:
			outputs = layers.Dense(units=15376, activation="softmax")(x)
		# Define the model.
		model = Model(inputs, outputs, name="3dcnn")
		return model


	def double_3d(self):

		inputs1 = keras.Input((self.width, self.height, self.depth, 1))
		inputs2 = keras.Input((self.width, self.height, self.depth, 1))

		x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs1)
		x1 = layers.MaxPool3D(pool_size=2)(x)

		x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x1)
		x2 = layers.MaxPool3D(pool_size=2)(x)

		x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x2)
		x3 = layers.MaxPool3D(pool_size=2)(x)

		x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x3)
		x4 = layers.MaxPool3D(pool_size=2)(x)

		x = layers.GlobalAveragePooling3D()(x4)
		x = layers.Dense(units=512, activation="relu")(x)
		x5 = layers.Dropout(self.do)(x)


		y = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs2)
		y1 = layers.MaxPool3D(pool_size=2)(y)
		y = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(y1)
		y2 = layers.MaxPool3D(pool_size=2)(y)

		y = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(y2)
		y3 = layers.MaxPool3D(pool_size=2)(y)

		y = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(y3)
		y4 = layers.MaxPool3D(pool_size=2)(y)

		y = layers.GlobalAveragePooling3D()(y4)
		y = layers.Dense(units=512, activation="relu")(y)
		y5 = layers.Dropout(self.do)(y)

		Rx1=layers.Flatten(name='flatten_tunedRx1')(x1)
		Ry1=layers.Flatten(name='flatten_tunedRy1')(y1)
		R1=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(Rx1,Ry1)

		Rx2=layers.Flatten(name='flatten_tunedRx2')(x2)
		Ry2=layers.Flatten(name='flatten_tunedRy2')(y2)
		R2=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(Rx2,Ry2)

		Rx3=layers.Flatten(name='flatten_tunedRx3')(x3)
		Ry3=layers.Flatten(name='flatten_tunedRy3')(y3)
		R3=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(Rx3,Ry3)


		Rx4=layers.Flatten(name='flatten_tunedRx4')(x4)
		Ry4=layers.Flatten(name='flatten_tunedRy4')(y4)
		R4=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(Rx4,Ry4)

		R=layers.Concatenate()([R1,R2,R3,R4])
		print(R.shape)
		R=tf.reshape(R, [1, 158898176, 1])
		rg1=layers.MaxPooling1D(pool_size=64)(R)
		rg1f=layers.Flatten(name='flatten_rg1')(rg1)                
		rg = layers.Dense(units=(4096), activation="relu")(rg1f)
		print(rg.shape)
		return Model(inputs=[inputs1,inputs2],outputs=rg,name="double_3D")


	def tune_MHL(self,backbone="none",name="",attention="_3d_image_classification",store_model="",parallel='off'):
		inputs=keras.Input((self.width,self.height,self.depth,1))
		if backbone=="none":
			x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)
			print("case M-Head attention MHL ")
			x = layers.GlobalAveragePooling3D()(x)
			rc = layers.Dense(units=(self.height*self.width), activation="relu")(x)
			
		elif backbone=="simple_3d_tune":
			Smodel=self.simple_3d('on')
			model_file=str(store_model + "/"+self.backb_w)
			print(model_file)
			if os.path.exists(model_file):
				Smodel.load_weights(model_file,by_name=True, skip_mismatch=True)
				print('load denset weights')
			rc=Smodel(inputs)
		elif backbone=="simple_3d":
			x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			rc1 = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			rc2 = layers.BatchNormalization()(x)

			x = layers.GlobalAveragePooling3D()(x)
			x = layers.Dense(units=512, activation="relu")(x)
			rc = layers.Dense(units=(15376), activation="relu")(x)
			print("case M-Head attention simple model ")
		else:
			print("No none backbone network try resnet50, densenet121, or none!")
		Rdo=layers.Flatten(name='flatten_tunedR')(rc)
		if parallel=='on':
			Rd1=layers.Flatten(name='flatten_tunedR1')(rc1)
			Rd2=layers.Flatten(name='flatten_tunedR2')(rc2)
			R=layers.MultiHeadAttention(num_heads=3,key_dim=self.height,attention_axes=(1))(Rd1,Rd2,Rdo)
			Rd=R
		else:
			Rd=Rdo
		xrgb=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(Rd,Rd)
		print(xrgb.shape)
		f=layers.Flatten(name='flatten_R')(xrgb)
		rgb = layers.Dense(units=(15376), activation="relu")(f)
		if parallel=='on':                
			rgb1=layers.Reshape([124,124,1,1])(rgb)
			rgb2=layers.Reshape([124,124,1,1])(rc)
			rgbc=layers.Concatenate(axis=3)([rgb1,rgb2])
			r=layers.Reshape([124,124,2,1])(rgbc)
			rgbo = layers.MaxPool3D(pool_size=(1,1,2))(r)
		else:
			rgbo=rgb
		rgb11=layers.Reshape([124,124,1,1])(rgbo)
		RCC=layers.Conv3D(filters=self.depth, kernel_size=1, activation="relu")(rgb11)
		rgx=layers.Reshape([124,124,self.depth,1])(RCC)
		x = layers.GlobalAveragePooling3D()(rgx)
		Rdx=layers.Flatten(name='flatten_tunedRx')(x)  
		rg = layers.Dense(units=1024, activation="relu")(Rdx)
		return Model(inputs, rg,name="3dmhl")


	def MLP(self,pretrained_model):
                
		new_DL=pretrained_model.output
		new_DL=layers.Flatten()(new_DL)
		new_DL=layers.Dense(1024, activation="relu")(new_DL)   #64
		new_DL=layers.Dropout(self.do)(new_DL)
		new_DL=layers.Dense(512, activation="relu")(new_DL)    #64
		new_DL=layers.Dropout(self.do)(new_DL)
		new_DL=layers.Dense(self.classes, activation="softmax")(new_DL) #2
		return Model(inputs=pretrained_model.input, outputs=new_DL)
