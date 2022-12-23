## Data Mining Project

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
 




## Cancer diagnose NN
checklist :

- Decide which of the columns we are going to use, using information gain algorithm or gradient decent.
- Design a neural network and train it
- Create a way to display the output - accuracy, recall... Find a python package the genarates Neural network image.
- Write a powerpoint slide

## Image Process
checklist:

- Learn about *transfer learning* and *dropout*
- Use **Keras** to impl. kernel analysis (strode , size , how much , Maxpool\Avrpool).
- Design a Nerual Network and train it. (how many aphocs? what size of batchs?)
- Create a way to display the output - accuracy, recall... Find a python package the genarates Neural network image.
- [Optional] Output the neural network thinking process.




