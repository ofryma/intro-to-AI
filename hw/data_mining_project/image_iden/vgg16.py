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



vgg16_model = vgg16.VGG16(weights='imagenet')
# vgg16_model.summary()

# Utility Function to Load Image, Preprocess input and Targets
def predict_image(model, img_path, preprocess_input_fn, decode_predictions_fn, target_size=(224, 224)):

    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_fn(x)
    
    preds = model.predict(x)
    predictions_df = pd.DataFrame(decode_predictions_fn(preds, top=10)[0])
    predictions_df.columns = ["Predicted Class", "Name", "Probability"]
    return predictions_df

#img_path="rocking_chair.png"  ## Uncomment this and put the path to your file here if desired
# Predict Results
print(predict_image(vgg16_model, img_path, vgg16.preprocess_input, vgg16.decode_predictions))

