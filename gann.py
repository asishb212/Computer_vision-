import tensorflow as tf
from tensorflow import keras
import numpy as np
import plot_utils
import matplotlib.pyplot as plt
import tqdm
print('Tensorflow version:', tf.__version__)

def show(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

pd=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(pd[0],True) 

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

batch_size=32
datas=tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
datasetf=datas.batch(batch_size,drop_remainder=True).prefetch(1)

#generator
num_features=100
generator=tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*256,input_shape=[num_features]), 
    tf.keras.layers.Reshape([7,7,256]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64,(5,5),(2,2),padding='same',activation='selu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(1,(5,5),(2,2),padding='same',activation='tanh'),
])
print(generator.summary())
noise=tf.random.normal(shape=[1,num_features])
generated_img=generator(noise,training=False)
show(generated_img)
plt.show()

descriminator=tf.keras.Sequential([
    tf.keras.layers.Conv2D(128,(5,5),(2,2),padding='same',input_shape=(28,28,1)),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64,(5,5),(2,2),padding='same'),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
decision=descriminator(generated_img,training=False)
print(decision)

#compiling
descriminator.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
descriminator.trainable=False

gan=tf.keras.models.Sequential([generator,descriminator])
gan.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

seed=tf.random.normal(shape=[batch_size,100])
#training
def train_dcgan(gan,dataset,batch_size,num_features,epochs=5):
    generator,descriminator=gan.layers
    for epoch in tqdm.tqdm(range(epochs)):
        print("[INFO] current epoch:",epoch)
        for x_batch in dataset:
            noise=tf.random.normal(shape=[batch_size,num_features])
            generated_images=generator(noise)
            x_fake_n_real=tf.concat([generated_images,x_batch],axis=0)
            y1=tf.constant([[0.]]*batch_size+[[1.]]*batch_size)
            descriminator.trainable=True
            descriminator.train_on_batch(x_fake_n_real,y1)
            y2=tf.constant([[1.]]*batch_size)
            descriminator.trainable=False
            gan.train_on_batch(noise,y2)
        generate_and_save_images(generator,epoch+1,seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(10,10))
    
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

#training
x_train_dcgan=x_train.reshape(-1,28,28,1)*2. -1. 
batch_size=32
dataset=tf.data.Dataset.from_tensor_slices(x_train_dcgan).shuffle(1000)
dataset=dataset.batch(batch_size,drop_remainder=True).prefetch(1)

train_dcgan(gan,dataset,batch_size,num_features,epochs=10)
gan.save('gan.h5')
descriminator.save('disc.h5')
generator.save('gen.h5')
