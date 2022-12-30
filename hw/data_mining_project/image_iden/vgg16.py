import os
import zipfile

from shutil import copy , rmtree



from PIL import Image

from keras.utils import load_img , img_to_array
import numpy as np
import pandas as pd
from keras.applications import vgg16
#from scipy.misc import imread
import matplotlib.pyplot as plt


def extract_data(zip_path : str , dest_path : str):
    # extract all the files
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(dest_path)
    zip_ref.close()

def arrange_by_tags(  tags_dict : dict , src_dir : str , filterd_dir : str , train_size : int , rearrange : bool = True):
    
    
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


    return tags_dict



wheater_tags = {
    'rain' : [] ,
    'shine' : [] ,
    'cloudy' : [] ,
    'sunrise' : [] ,
}


wheater_tags = arrange_by_tags(
    tags_dict = wheater_tags,
    filterd_dir = os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather_filtered"),
    train_size= 150,
    src_dir=os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather")
)





        
    












# vgg16_model = vgg16.VGG16(weights='imagenet')
# # vgg16_model.summary()

# # Utility Function to Load Image, Preprocess input and Targets
# def predict_image(model, img_path, preprocess_input_fn, decode_predictions_fn, target_size=(224, 224)):

#     img = load_img(img_path, target_size=target_size)
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input_fn(x)
    
#     preds = model.predict(x)
#     predictions_df = pd.DataFrame(decode_predictions_fn(preds, top=10)[0])
#     predictions_df.columns = ["Predicted Class", "Name", "Probability"]
#     return predictions_df

# #img_path="rocking_chair.png"  ## Uncomment this and put the path to your file here if desired
# # Predict Results
# print(predict_image(vgg16_model, img_path, vgg16.preprocess_input, vgg16.decode_predictions))

