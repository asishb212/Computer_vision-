import os 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.applications.resnet50 import ResNet50
import keras 
from tensorflow.keras.optimizers import RMSprop 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
base_dir="/home/ashifer/code/datasets/cats_and_dogs_filtered" 
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
#path="/home/ashifer/code/tensflow/_resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" 
trained_model=ResNet50(input_shape=(150,150,3),include_top=False,weights='imagenet') 
#trained_model.load_weights(path)
for layer in trained_model.layers: 
    layer.trainable=False 
trained_model.summary()
last_layer = trained_model.get_layer('conv5_block3_out')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
x=tf.keras.layers.Flatten()(last_output)
model=tf.keras.Model(trained_model.input)
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                   
x = layers.Dense  (1, activation='sigmoid')(x)           
model = tf.keras.Model(trained_model.input, x)  
model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=20,class_mode='binary',target_size=(150,150))
valid_generator=test_datagen.flow_from_directory(validation_dir,batch_size=20,class_mode='binary',target_size=(150,150))
history=model.fit(train_generator,steps_per_epoch=50,epochs=10,verbose=1,validation_data=valid_generator,validation_steps=50)
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy']
loss = history.history['loss'] 
val_loss = history.history['val_loss']  
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure() 
plt.show() 