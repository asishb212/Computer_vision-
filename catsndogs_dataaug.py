import os 
import tensorflow as tf 
import numpy as np 
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt  
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
base_dir="/home/asish/code/datasets/cats_and_dogs_filtered" 
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
model=tf.keras.Sequential() 
model.add(tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3))) 
model.add(tf.keras.layers.MaxPool2D((2,2))) 
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu')) 
model.add(tf.keras.layers.MaxPool2D((2,2)))  
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu')) 
model.add(tf.keras.layers.MaxPool2D((2,2))) 
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(512,activation='relu')) 
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=["accuracy"])
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=20,class_mode='binary',target_size=(150,150))
valid_generator=test_datagen.flow_from_directory(validation_dir,batch_size=20,class_mode='binary',target_size=(150,150))
history=model.fit(train_generator,steps_per_epoch=100,epochs=100,verbose=1,validation_data=valid_generator,validation_steps=50)
acc      = history.history['accuracy'] 
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss' ]
epochs   = range(len(acc)) 
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss')
plt.show()