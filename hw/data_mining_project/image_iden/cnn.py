import os
import zipfile
from shutil import copy , rmtree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from keras.utils import load_img , img_to_array
from keras import layers
from keras import Model
from keras.optimizers import RMSprop , SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
from keras.applications import vgg16
from keras.models import load_model

def extract_data(zip_path : str , dest_path : str = ""):
    # extract all the files
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(dest_path)
    zip_ref.close()

def arrange_by_tags( 
    tags_dict : dict , 
    src_dir : str , 
    filterd_dir : str , 
    train_size : int , 
    rearrange : bool = True , 
    show_data_length : bool = True):
    
    
    if os.path.exists(filterd_dir):
        rmtree(filterd_dir)

    os.mkdir(filterd_dir)
    os.mkdir(os.path.join(filterd_dir , "train"))
    os.mkdir(os.path.join(filterd_dir , "test"))
    

    img_file_list = os.listdir(src_dir)

    for img in img_file_list:
        for key in list(tags_dict.keys()):
            if key in img:
                tags_dict[key].append(img)


    train_images = {
        'rain' : wheater_tags['rain'][:train_size] ,
        'shine' : wheater_tags['shine'][:train_size] ,
        'cloudy' : wheater_tags['cloudy'][:train_size] ,
        'sunrise' : wheater_tags['sunrise'][:train_size] ,
    }
    test_images = {
        'rain' : wheater_tags['rain'][train_size:] ,
        'shine' : wheater_tags['shine'][train_size:] ,
        'cloudy' : wheater_tags['cloudy'][train_size:] ,
        'sunrise' : wheater_tags['sunrise'][train_size:] ,
    }

    for key in list(train_images.keys()):
        for filename in train_images[key]:

            if not os.path.exists(f"{filterd_dir}\\train\\{key}"):
                path = os.path.join(f"{filterd_dir}\\train", key)
                
                os.mkdir(path)

            copy(f"{src_dir}\\{filename}" , f"{filterd_dir}\\train\\{key}\\{filename}")
            
        for filename in test_images[key]:
            if not os.path.exists(f"{filterd_dir}/test/{key}"):
                os.mkdir(f"{filterd_dir}/test/{key}")

            copy(f"{src_dir}/{filename}" , f"{filterd_dir}/test/{key}/{filename}")

    if show_data_length:
        for key in list(tags_dict.keys()):
            print(f"list length of {key} : " , len(tags_dict[key]))

    return tags_dict

def create_model(
        output_options_number : int ,
        print_summary = True,
        image_shape = (150,150,3),
        output_activation : str = 'softmax',
        with_dropout : bool = False,
        ):
                
        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=image_shape)

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # if with_dropout:
        #         x = layers.Dropout()(x)

        # Flatten feature map to a 1-dim tensor so we can add fully connected layers
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(output_options_number , activation=output_activation)(x)

        # Create model:
        # input = input feature map
        # output = input feature map + stacked convolution/maxpooling layers + fully 
        # connected layer + sigmoid output layer
        model = Model(img_input, output)

        if print_summary:
                model.summary()
        
        model.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['acc'])


        return model

def create_image_gens(
        filterd_dir : str, 
        set_class_mode : str = 'categorical',
        image_shape = (150, 150)
):

        train_dir = os.path.join(filterd_dir , "train")
        test_dir = os.path.join(filterd_dir , "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
                return


        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Flow training images in batches of 20 using train_datagen generator
        train_gen = train_datagen.flow_from_directory(
                train_dir,  # This is the source directory for training images
                target_size=image_shape,  # All images will be resized to 150x150
                batch_size=20,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode=set_class_mode
                )


        # Flow validation images in batches of 20 using val_datagen generator
        test_gen = test_datagen.flow_from_directory(
                test_dir,
                target_size=image_shape,
                batch_size=20,
                class_mode=set_class_mode
                )

        return train_gen , test_gen

def print_model_history_data(history):
                
        # Retrieve a list of accuracy results on training and validation data
        # sets for each training epoch
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        # Retrieve a list of list results on training and validation data
        # sets for each training epoch
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Get number of epochs
        epochs = range(len(acc))

        # Plot training and validation accuracy per epoch
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Training and validation accuracy')

        plt.figure()

        # Plot training and validation loss per epoch
        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Training and validation loss')

def predict_image_with_vgg16(model, img_path, preprocess_input_fn, decode_predictions_fn, target_size=(224, 224)):

    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_fn(x)
    
    preds = model.predict(x)
    predictions_df = pd.DataFrame(decode_predictions_fn(preds, top=10)[0])
    predictions_df.columns = ["Predicted Class", "Name", "Probability"]
    return predictions_df

def predict_image_with_cnn(model, img_path , pred_key_list : list , input_shape = (150,150,3) ):

        pred_key_list.sort()

        img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
        x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

        # Rescale by 1/255
        x /= 255

        # Let's run our image through our network, thus obtaining all
        # intermediate representations for this image.
        successive_feature_maps = model.predict(x)

        pred_dict = {}

        for key , pred in zip(pred_key_list , successive_feature_maps[0]):
                pred_dict[key] = pred

        return pred_dict

def cnn_weater_model( f_d : str , s_d : str , rearrange : bool = True , model_path = None):


        model_path = os.path.join(os.getcwd() , model_path)
        if not model_path == None and os.path.exists(model_path):
                print("Loading model from  " , model_path)
                loaded_model = load_model(model_path)
                return loaded_model


        print("Creating new model...")
        epoch_number = 10
        image_target_size = (150,150,3)

        wheater_tags = {
                'shine' : [] ,
                'cloudy' : [] ,
                'sunrise' : [] ,
                'rain' : [] ,
        }
        if rearrange:
                 
  

                wheater_tags = arrange_by_tags(
                tags_dict = wheater_tags,
                filterd_dir = f_d,
                train_size= 150,
                src_dir=s_d
                )

        # dumb model
        our_model = create_model( output_options_number = len(wheater_tags) , output_activation= "softmax" , print_summary=False)
        
        # create generators
        train_generator , test_generator = create_image_gens(
                filterd_dir=f_d,
                image_shape =(150,150)
                )

        # training the model
        history = our_model.fit(
        train_generator,
        #steps_per_epoch=100,  # 2000 images = batch_size * steps
        epochs=epoch_number,
        validation_data=test_generator,
        #validation_steps=50,  # 1000 images = batch_size * steps
        verbose=2
        )

        new_model_path = './weater_model.h5'
        print("Saving new model to " , new_model_path )
        our_model.save(new_model_path)

        return our_model

filterd_dir = os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather_filtered")
source_dir = os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather")

weater_model = cnn_weater_model(f_d=filterd_dir , s_d=source_dir , rearrange=False , model_path="weater_model.h5")
vgg16_model = vgg16.VGG16(weights='imagenet')

prediction_keys = ["rain" , "cloudy" , "shine" , "sunrise"]

## Predictions

for i in range(3):
        print("*"*80)
        img_file_index = random.randint(0 , len(os.listdir(source_dir)))
        rand_name = os.listdir(source_dir)[img_file_index]
        img_path = os.path.join(source_dir , rand_name)
        print(rand_name)
        print(predict_image_with_cnn(weater_model, img_path , prediction_keys))
        # print(predict_image_with_vgg16(vgg16_model, img_path, vgg16.preprocess_input, vgg16.decode_predictions))
        print("*"*80)
