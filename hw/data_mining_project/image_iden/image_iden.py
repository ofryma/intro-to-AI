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
# from keras.applications import vgg16
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ModelUse():
        def __init__(
                self,
                prediction_keys : list, # list of the tags this model know
                data_src_dir : str, # source directory of the image data files
                filtered_dir : str, # destination directory of the image files
                train_split_size : int = None, # how many image going to the train data
                rearange_data : bool = True, # use only for the first time to transfer from the source to the destination dirs
                image_input_shape = (150,150,3),
                target_size = (150,150),
                output_activation : str = 'softmax',
                show_logs : bool = True, # manage the logs along the way
                use_dropout : bool = False, 
                optimizer = None, # the funciton that get send to the compile of the model
                learning_rate : float = 0.001,
                train_batch_size : int = 20,
                test_batch_size : int = 20,
                existing_model_path = None, # is there is a model in a .h5 file, we can load it instead of creating NN again
                epoch_number : int = 10, 
                verbose : int = 2, # logs of the apoch proccess
                model_name : str = "my_model", # the name of the new modle that saved ( my_model.h5 )
                
                
        ):
                
                self.verbose = verbose
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
                self.use_dropout = use_dropout
                self.learning_rate = learning_rate
                if optimizer == None:
                        self.use_optimizer = RMSprop(learning_rate=learning_rate)
                else:
                        self.use_optimizer = optimizer
                self.epoch_number = epoch_number
                self.data_src_dir = data_src_dir
                self.filtered_dir = filtered_dir
                self.train_split_size = train_split_size
                self.image_target_size = target_size # i.e. 150X150 (150,150)
                self.train_batch_size = train_batch_size
                self.test_batch_size = test_batch_size
                self.image_class_mode = 'categorical'
                self.verbose = verbose

                self.model_name = model_name
                self.model = self.cnn_model()

        def create_image_gens(self , rescale_factor = 1./255):

                train_dir = os.path.join(self.filtered_dir , "train")
                test_dir = os.path.join(self.filtered_dir , "test")

                if not os.path.exists(train_dir) or not os.path.exists(test_dir):
                        return


                # All images will be rescaled by 1./255
                train_datagen = ImageDataGenerator(rescale=rescale_factor)
                test_datagen = ImageDataGenerator(rescale=rescale_factor)

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
                output = layers.Dense(len(self.prediction_keys) , activation=self.output_activation)(x)

                # Create model:
                # input = input feature map
                # output = input feature map + stacked convolution/maxpooling layers + fully 
                # connected layer + sigmoid output layer
                model = Model(img_input, output)

                if self.show_log:
                        model.summary()

                model.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['acc'])

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
                
                img = load_img(img_path, target_size=self.image_target_size)  # this is a PIL image
                x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
                x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

                # Rescale by 1/255
                x /= 255

                # Let's run our image through our network, thus obtaining all
                # intermediate representations for this image.
                successive_feature_maps = model.predict(x)

                pred_dict = {}

                for key , pred in zip(self.prediction_keys , successive_feature_maps[0]):
                        pred_dict[key] = pred
                

                return pred_dict

        def cnn_model(self): 
                if not self.existing_model_path == None and os.path.exists(self.existing_model_path):
                        print("Loading model from  " , self.existing_model_path)
                        self.model = load_model(self.existing_model_path)
                        return self.model
                        
                        
                print("Creating new model...")
                              
                if self.rearange_data:
                        self.arrange_by_tags()

                # dumb model
                our_model = self.genarate_model()
                
                # create generators
                train_generator , test_generator = self.create_image_gens()

                # training the model
                self.history = our_model.fit(
                        train_generator,
                        epochs=self.epoch_number,
                        validation_data=test_generator,
                        verbose=self.verbose
                )

                new_model_path = os.path.join(os.getcwd() , f'{self.model_name}.h5')
                try:
                        print("Saving new model to " , new_model_path )
                        our_model.save(new_model_path)
                        self.existing_model_path = new_model_path
                                
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

weater_model = ModelUse(
        data_src_dir=os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather"),
        filtered_dir=os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather_filtered"),
        prediction_keys = ["rain" , "cloudy" , "shine" , "sunrise"],
        existing_model_path="my_model.h5"
        )



# vgg16_model = vgg16.VGG16(weights='imagenet')


## Predictions

for i in range(3):
        print("*"*80)
        img_file_index = random.randint(0 , len(os.listdir(weater_model.data_src_dir)))
        rand_name = os.listdir(weater_model.data_src_dir)[img_file_index]
        img_path = os.path.join(weater_model.data_src_dir , rand_name)
        print(rand_name)
        results = weater_model.predict_image_with_cnn(img_path=img_path)
        print(results)
        # print(predict_image_with_vgg16(vgg16_model, img_path, vgg16.preprocess_input, vgg16.decode_predictions))
        print("*"*80)
