# Data Mining Project

There are many great datasets are out there and if you don’t like these options feel free to check out others [here](https://www.kdnuggets.com/datasets/index.html).


Each of the following datasets have different challenges.

1.	The first is a [cancer dataset](https://drive.google.com/file/d/1mwixo3EgQmxXCoFi_jdaEGOF84_tHfmL/view?usp=share_link) similar to the ones we have studied in class.

In this dataset there are 120 genomic markers for esophageal cancer. There are then *two* validation datasets—test and validation. The validation dataset is the “test” one used to run parameters in the model like how deep the decision tree, what attributes to use and how deep a neural network to use, and the test is the “real” test. How good a model did you get?  What features and models did you use? In order to answer, not only is model accuracy important, but feature analysis too.

2.	**Image Processing**.  In this problem you have four categories of weather:

* Cloudy  (מעונן)
* Rain  (גשום)
* Shine  (בהיר)
* Sunrise (זריחה)


You can download the files from [here](https://drive.google.com/file/d/10t_d-Rxl3TdeLXv8ey3hRUbBTuQKTy5W/view?usp=sharing)

What is the accuracy with and without **transfer learning**?  Did **Dropout** help? 

### Project Submission Directions:
Step 1: Decide if you want to work by yourself or with a partner. If you work with a partner you will need to do *both* assignments, but if you work by yourself you only need to pick one topic.   Please write your decision, and which project you want to do if you are working by yourself at link by December 28th:
https://docs.google.com/spreadsheets/d/1j8YTmhG2Rcs6fdb7yCyqIgtUaqC5aDZStF2Xgx6k1fc/edit?usp=sharing 

Step 2: Get to work!  You will need to work in Python, with a preference to working in notebooks (Colab or Jupyter) and not locally (Spyder,  Pycharm, etc.).
Note: When submitting, if submission is done by colab send a Word file with a link to the notebook collab, if submission is done by Jupyter, please provide an HTML file of the notebook with all the output.
Step 3: Submit and present your work!  You will need to present your work on the last day of class (January 18th).  Unless there is a special reason (e.g. Miluim) you will lose points if you do not present on that day.
In general, there are three major items you need to consider: Data analysis and setup (before you can create the machine learning model), the creation of the machine learning model, and presenting that model to someone who must decide if the results are interesting, overfitted or wrong. Here are some things to consider for each of these points:
1.	Data Setup
2.	For project #1 (genomic): Did you use any feature selection?  Did you create or find any features that were helpful for your project?
2.	For project #2 (images): Did you use cross validation or just one train / test split? In general, cross validation is better.
2.	Machine Model Logic
1.	What machine models did you try and which one worked the best?  For all solutions you must check a neural network with Keras.  What model architecture did you select?  How many nodes and layers?  For all projects other than the image processing one, also check simpler models we discussed in class such as regression (linear or logistic) and random forests.
2.	What measurements did you try to evaluate your success?  (Accuracy, Recall, AUC, recall, etc.)
3.	Presentation
1.	What graphs can you provide to help make your results easier to understand?  AUC graphs, result tables and graphical ways to view the important features are good ways to better understand the results
2.	Code comments are very important. Please document your code.
3.	A short word document of 2-3 pages or PowerPoint slide of up to 10 slides can help make your results clear.  Please do one or the other (Word document or PowerPoint).
Some resources:

Example for feature selection:
https://colab.research.google.com/drive/1ArYAcpSMiYEDe08uXh1SYyeo-uMEURFf?usp=sharing 
Example for train / test split similar to knn homework:
https://colab.research.google.com/drive/1yIKemvRF8pxEvxSsR8lSl_tf1MO68-sk?usp=sharing 
Example for cross validation for image processing:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md 


### Grade breakdown:
10 points for data setup
45 points for machine model logic
20 points for code correctness
25 points for presentation including code documentation
Total: 100 points  
 


> ## Cancer Diagnose
Inside the [cancer_diagnose](./cancer_dignose/) directory is the part of the project that implaments a neural network model created from a cancer dataset.

## [cancer_diagnose.py](./cancer_dignose/cancer_diagnose.py) algorithem :
1. Split the data - the data set is splited to three files - 
    - [train.csv](./cancer_dignose/train.csv)
    - [testing.csv](./cancer_dignose/testing.csv)
    - [validation.csv](./cancer_dignose/validation.csv)

The first thing we need to do is split the data to X (data) and y (targes) for each of them (CSVs), clean the data (remove extra columns etc.).

2. Feature Selection -
3. Normalize the data - 
4. Generate the model - in this section we created a model with the configuration as shown in the code. The model is created inside the `genarate_model()` function. Then we use the `compile` method on the model.
5. Search for a good fit - We created a funciton that using the `fit` method number of times and finds the best model fit with the best accuracy out of those tries.
6. Use the model and desplay the learning proccess - 


> ## Weater Recognition - Image Indetification

All this part of the project is implamented in the [image_iden](./image_iden/image_iden.py) script and splits to two sections:

> Creating CNN from scratch

1. Organize the data - We extracted the data from the dataset zip that was given into [weater](./image_iden/weather/) folder. all the images are sitting in this folder without any sorting at all. Then we built a function called `arrange_by_tags` that copy the images in a smart way to a folder called [weather_filtered](./image_iden/weather_filtered/).
2. Creating the Nerual Network Architecture - now we are creating a new model with out special configurations.
3. Create ImageDataGenerators - We used the `ImageDataGenerator` class to create a flow of images from the [weather_filtered](./image_iden/weather_filtered/). This flow is later sent to the `fit` method of this model.
4. Teach the model - Sending the image flow to the `model.fit` nethod.
5. Make a prediction - We created a funciton called `predict_image_with_cnn` that will output a dictionary of the tags and the probability for that tag in relation to the image that sent to it, using the `model.predict` method.

> Using VGG16 - transfer learning








