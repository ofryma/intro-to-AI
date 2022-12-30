import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras import layers
from keras import activations

def split_data_target( dataframe , target_column : str ):

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    return X , y

def convert_from_str_to_int(string_list : list , conversion_dict : dict):

    """
        This function takes a convertion dictionary : { "Famale" : 1 , "Male" : 0}
        and for every element in the string_list, converts it according to this dictionaty
    """

    new_list = []
    for item in string_list:
        new_list.append(conversion_dict[item])
    
    return new_list

def df_to_list(df):
    # Convert the dataframe to a NumPy array
    df_keys = list(df.keys())

    list_of_lists = []


    for i in range(len(df)):
        cur_list = []
        for k in df_keys:
            cur_list.append(df[k][i])
        
        list_of_lists.append(cur_list)
    
    return list_of_lists

def genarate_model( 
    n_list : list , 
    activations_list : list ,
    input_dim : int , 
    input_activation_function : str = "relu" ,
    output_node_number : int = 1,
    output_activation_function : str = "sigmoid",
    model_visualization : bool = False, 
    model_summary : bool = False, 
    first_hidden_layer : int = 50,
    ):

    # Define the model architecture
    model = Sequential()

    model.add(Dense(50 , input_dim=input_dim, activation=input_activation_function))

    for node , act_function in zip(n_list , activations_list):
        model.add(Dense(node, activation=act_function))
        
    
        
    model.add(Dense(output_node_number, activation=output_activation_function))


    if model_summary:
        print(model.summary())
    if model_visualization:
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



    return model

def find_best_model_fit(model , num_of_iters : int = 1):
    
    best_model = None
    best_acc = 0

    acc_list = []
    for _ in range(num_of_iters):
            
        # train the model (using the validation set)
        model.fit(X_train_norm , y_train , validation_data=(X_validation_norm , y_validation) , epochs = epoch_number , verbose=0)

        # making a prediction
        y_pred_prob_nn_1 = model.predict(X_test_norm)
        y_pred_class_nn_1 = np.rint(y_pred_prob_nn_1)


        cur_acc = accuracy_score(y_test,y_pred_class_nn_1)


        acc_list.append(cur_acc)
        
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model = model


    print(best_acc)
    return best_model


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("testing.csv")
validation_data = pd.read_csv("validation.csv")

# aplit the data to X and y
X_train , y_train = split_data_target(train_data , "Diagnosis")
X_test , y_test = split_data_target(test_data , "Diagnosis")
X_validation , y_validation = split_data_target(validation_data , "Diagnosis")

# convert the string columns into integer columns
X_train["gender"] = convert_from_str_to_int(X_train["gender"] , {"Female" : 1 , "Male" : 0})
y_train = convert_from_str_to_int(y_train , {"NonCancer" : 0 , "Cancer" : 1})
X_test["gender"] = convert_from_str_to_int(X_test["gender"] , {"Female" : 1 , "Male" : 0})
y_test = convert_from_str_to_int(y_test , {"NonCancer" : 0 , "Cancer" : 1})
X_validation["gender"] = convert_from_str_to_int(X_validation["gender"] , {"Female" : 1 , "Male" : 0})
X_validation = X_validation.drop(columns=["PtNo"])
y_validation = convert_from_str_to_int(y_validation , {"NonCancer" : 0 , "Cancer" : 1})


# converting the dataframes into list of all the rows in the dataframe


normalizer = StandardScaler()


X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)
X_validation_norm = normalizer.transform(X_validation)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_validation = np.asarray(y_validation)




# variables
input_dim = len(list(X_train.keys()))
epoch_number = 100
n_list = [100 , 120] # number of nodes in each hidden layer
activations_list = ["relu" , "relu" , "relu"] # activation function in each hidden layer
learning_rate = 0.003
momentum = 0.3
find_model_tries = 4 # nmber of times to run the model.fit before getting back the best one


model = genarate_model(
    n_list=n_list,
    activations_list=activations_list,
    input_dim=input_dim,
    model_summary = False,
)

# Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(SGD(lr = learning_rate, momentum=momentum), loss='binary_crossentropy', metrics=['accuracy'])

# find the best model
model = find_best_model_fit(model , find_model_tries)
















