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
from keras.applications import VGG16
from keras.models import load_model



def extract_data(zip_path : str , dest_path : str = ""):
    # extract all the files
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(dest_path)
    zip_ref.close()

def predict_image_with_vgg16(model, img_path, preprocess_input_fn, decode_predictions_fn, target_size=(224, 224)):

    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_fn(x)
    
    preds = model.predict(x)
    predictions_df = pd.DataFrame(decode_predictions_fn(preds, top=10)[0])
    predictions_df.columns = ["Predicted Class", "Name", "Probability"]
    return predictions_df
	

class ModelUse():
        def __init__(
                self,
                prediction_keys : list, # list of the tags this model know
                filtered_dir : str,
                data_src_dir : str = None, # source directory of the image data files
                 # destination directory of the image files
                train_split_size : int = None, # how many image going to the train data
                rearange_data : bool = True, # use only for the first time to transfer from the source to the destination dirs
                image_input_shape = (150,150,3),
                target_size = (150,150),
                output_activation : str = 'softmax',
                show_logs : bool = True, # manage the logs along the way

                # neural network settings
                k_size : int = 3,
                conv2_act_func : list = ["relu" , "relu" , "relu"],
                conv2_node_number : list = [16 , 32 , 64],


                use_dropout : bool = False, # decide if you want to add dropout layer
                dropout_rate : float = 0.5, # if there is a dropout layer, decide the dropout rate

                optimizer = None, # the funciton that get send to the compile of the model
                learning_rate : float = 0.001,
                loss_function : any = 'binary_crossentropy',
                train_batch_size : int = 20,
                test_batch_size : int = 20,
                image_class_mode : str = 'categorical',
                existing_model_path = None, # is there is a model in a .h5 file, we can load it instead of creating NN again
                epoch_number : int = 10, 
                verbose : int = 2, # logs of the apoch proccess

                model_name : str = "my_model", # the name of the new modle that saved ( my_model.h5 )
                
                
        ):
                
                try:
                        self.existing_model_path = os.path.join(os.getcwd() , existing_model_path)
                except:
                        self.existing_model_path = None
                self.rearange_data = rearange_data
                prediction_keys.sort()
                self.prediction_keys = prediction_keys
                self.data_dict = {}
                for key in self.prediction_keys:
                        self.data_dict[key] = []
                self.image_input_shape = image_input_shape
                self.output_activation = output_activation
                self.show_log = show_logs

                self.k_size = k_size
                self.conv2_act_function = conv2_act_func
                self.conv2_node_number = conv2_node_number

                self.use_dropout = use_dropout
                self.dropout_rate = dropout_rate

                self.learning_rate = learning_rate
                if optimizer == None:
                        self.use_optimizer = RMSprop(learning_rate=learning_rate)
                else:
                        self.use_optimizer = optimizer
                self.loss_function = loss_function
                self.epoch_number = epoch_number
                self.data_src_dir = data_src_dir
                self.filtered_dir = filtered_dir
                self.train_split_size = train_split_size
                self.image_target_size = target_size # i.e. 150X150 (150,150)
                self.train_batch_size = train_batch_size
                self.test_batch_size = test_batch_size
                self.image_class_mode = image_class_mode
                self.verbose = verbose

                self.model_name = model_name
                self.model = self.cnn_model()

                try:
                        self.vgg_tl_model = self.vgg16_transfer_learning_model()
                except Exception as e:
                        print("Something went wrong with building the vgg model" ) # , e


        def create_image_gens(self , rescale_factor = 1./255 , datagen_mean : any = False):

                train_dir = os.path.join(self.filtered_dir , "train")
                test_dir = os.path.join(self.filtered_dir , "test")

                if not os.path.exists(train_dir) or not os.path.exists(test_dir):
                        return


                # All images will be rescaled by 1./255
                train_datagen = ImageDataGenerator(rescale=rescale_factor)
                test_datagen = ImageDataGenerator(rescale=rescale_factor)


                if datagen_mean:
                        train_datagen.mean = datagen_mean
                        test_datagen.mean = datagen_mean


                # Flow training images in batches of 20 using train_datagen generator
                train_gen = train_datagen.flow_from_directory(
                        train_dir,  # This is the source directory for training images
                        target_size=self.image_target_size,  # All images will be resized to 150x150
                        batch_size=self.train_batch_size,
                        # Since we use binary_crossentropy loss, we need binary labels
                        class_mode=self.image_class_mode
                        )


                # Flow validation images in batches of 20 using val_datagen generator
                test_gen = test_datagen.flow_from_directory(
                        test_dir,
                        target_size=self.image_target_size,
                        batch_size=self.test_batch_size,
                        class_mode=self.image_class_mode
                        )

                return train_gen , test_gen

        def genarate_model(self):
                        
                # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
                # the three color channels: R, G, and B
                img_input = layers.Input(shape=self.image_input_shape)

                first = True
                # This loop use two lists of values to create the layers in this model
                # in a dynamic way, by adding Conv2D layer and MaxPool layer according to 
                # current activation function and number of nodes given in the lists
                for act_func , node_number in zip(self.conv2_act_function , self.conv2_node_number):
                        if first:        
                                x = layers.Conv2D(node_number, self.k_size, activation=act_func)(img_input)
                                first = False
                        else:
                                x = layers.Conv2D(node_number, self.k_size, activation=act_func)(x)                
                        x = layers.MaxPooling2D(2)(x)
                        

                # Flatten feature map to a 1-dim tensor so we can add fully connected layers
                x = layers.Flatten()(x)

                # Create a fully connected layer with ReLU activation and 512 hidden units
                x = layers.Dense(512, activation='relu')(x)

                if self.use_dropout:
                        x = layers.Dropout(self.dropout_rate)(x)
                
                
                if self.output_activation == "sigmoid":
                        output_lenght = 1
                elif self.output_activation == "softmax":
                        output_lenght = len(self.prediction_keys)
                
                output = layers.Dense(len(self.prediction_keys) , activation=self.output_activation)(x)

                # Create model:
                # input = input feature map
                # output = input feature map + stacked convolution/maxpooling layers + fully 
                # connected layer + sigmoid output layer
                model = Model(img_input, output)

                if self.show_log:
                        model.summary()

                model.compile(loss=self.loss_function ,optimizer=self.use_optimizer , metrics=['acc'])

                self.model = model
                return model

        def arrange_by_tags(self):
        
                if os.path.exists(self.filtered_dir):
                        rmtree(self.filtered_dir)

                os.mkdir(self.filtered_dir)
                os.mkdir(os.path.join(self.filtered_dir , "train"))
                os.mkdir(os.path.join(self.filtered_dir , "test"))
                

                img_file_list = os.listdir(self.data_src_dir)

                for img in img_file_list:
                        for key in list(self.data_dict.keys()):
                                if key in img:
                                        self.data_dict[key].append(img)


                train_images = {}
                test_images = {}

                if self.train_split_size == None:
                        # half of the size of the min length
                        length_of_data_list = [len(self.data_dict[key]) for key in list(self.data_dict.keys())]
                        self.train_split_size = round(min(length_of_data_list)/2)



                for key in list(self.data_dict.keys()):
                        train_images[key] = self.data_dict[key][:self.train_split_size]
                        test_images[key] = self.data_dict[key][self.train_split_size:]


                # copy the files from the source dir to the filtered dir
                for key in list(train_images.keys()):

                        # copy all the train files from the source dir to the filtered dir
                        for filename in train_images[key]:

                                if not os.path.exists(f"{self.filtered_dir}\\train\\{key}"):
                                        path = os.path.join(f"{self.filtered_dir}\\train", key)
                                        
                                        os.mkdir(path)

                                copy(f"{self.data_src_dir}\\{filename}" , f"{self.filtered_dir}\\train\\{key}\\{filename}")
                        
                        # copy all the test files from the source dir to the filtered dir
                        for filename in test_images[key]:
                                if not os.path.exists(f"{self.filtered_dir}/test/{key}"):
                                        os.mkdir(f"{self.filtered_dir}/test/{key}")

                                copy(f"{self.data_src_dir}/{filename}" , f"{self.filtered_dir}/test/{key}/{filename}")

                # print the length of all the files for a certain tag
                if self.show_log:
                        for key in list(self.data_dict.keys()):
                                print(f"list length of {key} : " , len(self.data_dict[key]))
                 
        def predict_image_with_cnn(self ,  img_path  , model = None ):

                if model==None:
                        model = self.model
                
                img = load_img(img_path, target_size=self.image_target_size)  
                x = img_to_array(img)  
                x = x.reshape((1,) + x.shape)

                # Rescale by 1/255
                x /= 255

                # Let's run our image through our network, thus obtaining all
                # intermediate representations for this image.
                successive_feature_maps = model.predict(x)
                print(successive_feature_maps)

                pred_dict = {}

                for key , pred in zip(self.prediction_keys , successive_feature_maps[0]):
                        pred_dict[key] = pred
                

                return pred_dict

        def predict_image_with_vgg(self, image_path , model = None):
                pass

        def cnn_model(self): 

                model_filename = os.path.join(os.getcwd() , ".".join([self.model_name , "h5"])) 
                if os.path.exists(model_filename):
                        print("Loading model from  " , model_filename)
                        self.model = load_model(model_filename)
                        return self.model
                        
                        
                print(f"Creating new model at {model_filename}...")


                if self.rearange_data:
                        self.arrange_by_tags()

                # dumb model
                our_model = self.genarate_model()
                
                # create generators
                train_generator , test_generator = self.create_image_gens()

                # training the model
                # self.history = 
                our_model.fit(
                        train_generator,
                        epochs=self.epoch_number,
                        validation_data=test_generator,
                        verbose=self.verbose
                )

                
                try:
                        print("Saving new model to " , model_filename )
                        our_model.save(model_filename)
                        self.existing_model_path = model_filename
                                
                except Exception as e:
                        print(e)

        
                self.model = our_model
                return our_model
                      
        def print_model_history_data(self):
                        
                # Retrieve a list of accuracy results on training and validation data
                # sets for each training epoch
                acc = self.history.history['acc']
                val_acc = self.history.history['val_acc']

                # Retrieve a list of list results on training and validation data
                # sets for each training epoch
                loss = self.history.history['loss']
                val_loss = self.history.history['val_loss']

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

        def vgg16_transfer_learning_model(self):
                # load model
                model = VGG16(include_top=False, input_shape=(224, 224, 3))
                # mark loaded layers as not trainable

                for layer in model.layers:
                        layer.trainable = False


                # add new classifier layers
                flat1 = layers.Flatten()(model.layers[-1].output)
                class1 = layers.Dense(128, activation=self.output_activation, kernel_initializer='he_uniform')(flat1)
                output = layers.Dense(1, activation=self.output_activation)(class1)
                
                # define new model
                model = Model(inputs=model.inputs, outputs=output)

                # compile model
                model.compile(optimizer=self.use_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                train_generator , test_generator = self.create_image_gens(datagen_mean=[123.68, 116.779, 103.939])

                model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=self.epoch_number, verbose=self.verbose)

                self.vgg_tl_model = model


                return model








