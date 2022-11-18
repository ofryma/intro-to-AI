from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score , f1_score


import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import pandas as pd
from matplotlib import pyplot as plt


def hw1(dataset , cv=10 , md = 80):

    # creating new tree cls
    cls = DecisionTreeClassifier()
    cls.max_depth = md

    # creating a model
    cls.fit(dataset.data , dataset.target)

    # clac accuracy
    ac =  cross_val_score(
        cls,
        dataset.data,
        dataset.target,
        scoring='accuracy',
        cv=cv
    )

    # clac precision
    pre =  cross_val_score(
        cls,
        dataset.data,
        dataset.target,
        scoring='precision_weighted',
        cv=cv
    )

    y_prd = cls.predict(dataset.data)

    rc = recall_score(dataset.target , y_prd ,average = "weighted", zero_division=0)


    return round(ac.mean() , 3) , round(pre.mean() , 3) , cls.tree_.max_depth , rc , f1_score(dataset.target , y_prd , zero_division=0, average = "macro")
   
def my_plot(x_label , y_label , table):    

    plt.plot(table[x_label] , table[y_label])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.show()

def main(dataset):

    table = {"Accuracy":[] , "Precision": [], "Max Depth": [], "Recall Score": [] , "F1 score" : [] }

    for i in range(1,11):
        
        v_list = hw1(dataset , md = i)

        table["Accuracy"].append(v_list[0])
        table["Precision"].append(v_list[1])
        table["Max Depth"].append(v_list[2])
        table["Recall Score"].append(v_list[3])
        table["F1 score"].append(v_list[4])

    df = pd.DataFrame(table)

    print(df.to_string(index=False))


    my_plot("Max Depth" , "Accuracy" , table)
    my_plot("Max Depth" , "Precision" , table)
    my_plot("Max Depth" , "Recall Score" , table)
    my_plot("Max Depth" , "F1 score" , table)


    best_ac = max(table["Accuracy"])
    best_ac_loc = table["Accuracy"].index(best_ac)
    best_md = table["Max Depth"][best_ac_loc]


    return best_md


dataset_list = [datasets.load_iris() , datasets.load_wine() , datasets.load_digits()]


for ds in dataset_list:
    bmd = main(ds)
    print("Best Max Depth: " , bmd)

    








    











