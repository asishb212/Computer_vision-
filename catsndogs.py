import os 
import tensorflow as tf 
import numpy as np 
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt  
tf.config.experimental.list_physical_devices('GPU') 
base_dir="/home/ashifer/code/datasets/cats_and_dogs_filtered" 
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
train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=20,class_mode='binary',target_size=(150,150))
valid_generator=test_datagen.flow_from_directory(validation_dir,batch_size=20,class_mode='binary',target_size=(150,150))
history=model.fit(train_generator,steps_per_epoch=50,epochs=3,verbose=1,validation_data=valid_generator,validation_steps=50)
#prediction
path="/home/ashifer/code/datasets/cats_and_dogs_filtered/validation/cats/cat.2056.jpg"
img=image.load_img(path, target_size=(150, 150))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images)
print(classes[0])
if classes[0]>0:
    print(" is a dog")
else:
    print(" is a cat")
    """
opts=[layer.output for layer in model.layers[1:]]
visualization_model=tf.keras.models.Model(inputs=model.input,outputs=opts)
img_path="/home/ashifer/code/datasets/cats_and_dogs_filtered/validation/cats/cat.2056.jpg"
img_in=load_img(img_path,target_size=(150,150))
x=img_to_array(img)
x=x.reshape((1,)+x.shape)
x/=255.0
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features) 
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x 
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
    """
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss')
plt.show()