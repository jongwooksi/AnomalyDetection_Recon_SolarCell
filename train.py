import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False

def unet():
    input_shape=(128,128,3)
    n_channels = input_shape[-1]
    inputs = tf.keras.layers.Input(input_shape)

    conv1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = tf.keras.layers.Dropout(0.2)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = tf.keras.layers.Dropout(0.2)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = tf.keras.layers.Dropout(0.2)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
   
    conv4 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.2)(conv4)
    
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5')(conv5)
    drop5 = tf.keras.layers.Dropout(0.2)(conv5)

    up6 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = tf.keras.layers.Dropout(0.2)(conv6)

    up7 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop6))
    merge7 = tf.keras.layers.concatenate([drop3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop7 = tf.keras.layers.Dropout(0.2)(conv7)

    up8 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop7))
    merge8 = tf.keras.layers.concatenate([drop2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    drop8 = tf.keras.layers.Dropout(0.2)(conv8)

    up9 = tf.keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop8))
    merge9 = tf.keras.layers.concatenate([drop1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(n_channels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
 
    model = tf.keras.Model(inputs, conv9)

    return model


if __name__ == "__main__" :

    TRAIN_PATH = './newtrain/'

    x_train = []
    y_train = []

    folder_list = os.listdir(TRAIN_PATH)
    
    for folder in folder_list:
        file_list = os.listdir(TRAIN_PATH + folder)
        for filename in file_list:
            filepath = TRAIN_PATH + folder + '/' + filename
            image = imread(filepath)
            image.tolist()
            image = image / 255.
            x_train.append(image)

    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True)
   
    x_train = np.array(x_train)

    train_datagen.fit(x_train)

    print(train_datagen)
    model = unet()
    x_train = x_train.reshape(-1,128,128,3)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4), loss='mean_absolute_error')

    hist = model.fit_generator(train_datagen.flow(x_train, x_train, batch_size=16), epochs=50000)
   
    model.save('solarModel2')
