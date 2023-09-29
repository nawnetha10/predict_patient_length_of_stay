import pandas as pd
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import webbrowser

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

main = tkinter.Tk()
main.title("A Two-Stage Model to Predict Surgical Patient's Lengths of Stay From an Electronic Patient Database")
main.geometry("1300x1200")

global filename, dataset
global X, Y
global error
global le1, le2, le3
global X_train, X_test, y_train, y_test, classifier

def upload():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename,nrows=1000)
    text.insert(END,str(dataset.head()))

def processDataset():
    global dataset
    global X, Y
    global le1, le2, le3
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    dataset.fillna(0, inplace = True)
    dataset.drop(['vdate'], axis = 1,inplace=True)
    dataset.drop(['discharged'], axis = 1,inplace=True)
    dataset['rcount'] = pd.Series(le1.fit_transform(dataset['rcount'].astype(str)))
    dataset['gender'] = pd.Series(le2.fit_transform(dataset['gender'].astype(str)))
    dataset['facid'] = pd.Series(le3.fit_transform(dataset['facid'].astype(str)))
    text.insert(END,str(dataset)+"\n\n")

    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset train & test split details\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing : "+str(X_test.shape[0])+"\n") 


def stageFirstError(model):
    scr = ['accuracy']
    stage1_error = cross_validate(model, X, Y, cv=10, scoring=scr, return_train_score=True)
    stage1_error = stage1_error.get("test_accuracy")
    stage1_error = 1 - np.mean(stage1_error)
    return stage1_error

def stageSecondError(model):
    model.fit(X_train, y_train)
    predict = cross_val_score(model, X_train, y_train, cv = 10)
    second_stage_error = np.mean(predict)
    return second_stage_error

def runCart():
    global X, Y, X_train, X_test, y_train, y_test, classifier, error
    error = []
    text.delete('1.0', END)
    rf = RandomForestClassifier()
    rf_stage1_error = stageFirstError(rf)
    error.append(rf_stage1_error)
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn_stage1_error = stageFirstError(knn)
    error.append(knn_stage1_error)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    classifier = dt
    cart_stage1_error = stageFirstError(dt)
    error.append(cart_stage1_error)

    rf_stage2_error = stageFirstError(rf)
    error.append(rf_stage2_error)
    knn_stage2_error = stageFirstError(knn)
    error.append(knn_stage2_error)
    dt = DecisionTreeRegressor(random_state=0)
    cart_stage2_error = stageFirstError(dt)
    error.append(cart_stage2_error)

    output = '<table border=1 align=center>'
    output+= '<tr><th>Dataset Size</th><th>Algorithm Name</th><th>Stage1 Error</th><th>Stage2 Error</th></tr>'
    #output+='<tr><td>'+str(X.shape[0])+'</td><td>Random Forest</td><td>'+str(rf_stage1_error)+'</td><td>'+str(rf_stage2_error)+'</td></tr>'
    output+='<tr><td>'+str(X.shape[0])+'</td><td>KNN</td><td>'+str(knn_stage1_error)+'</td><td>'+str(knn_stage1_error)+'</td></tr>'
    output+='<tr><td>'+str(X.shape[0])+'</td><td>Two-Stage CART</td><td>'+str(cart_stage1_error)+'</td><td>'+str(cart_stage2_error)+'</td></tr>'
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   

def predict():
    global classifier
    global le1, le2, le3
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    test.drop(['vdate'], axis = 1,inplace=True)
    test.drop(['discharged'], axis = 1,inplace=True)
    test['rcount'] = pd.Series(le1.transform(test['rcount'].astype(str)))
    test['gender'] = pd.Series(le2.transform(test['gender'].astype(str)))
    test['facid'] = pd.Series(le3.transform(test['facid'].astype(str)))
    test = test.values
    predict = classifier.predict(test)
    for i in range(len(predict)):
        text.insert(END,"Patient Test Data = "+str(test[i])+" ====> LENGTH of STAY PREDICTED AS "+str(predict[i])+" Days\n\n") 

def graph():
    df = pd.DataFrame([['KNN','Stage 1 Error',error[1]],['KNN','Stage 2 Error',error[4]],
                       ['CART','Stage 1 Error',error[2]],['CART','Stage 2 Error',error[5]],                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='A Two-Stage Model to Predict Surgical Patients Lengths of Stay From an Electronic Patient Database')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Patient Stay Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cartButton = Button(main, text="Run Two-State CART Algorithm", command=runCart)
cartButton.place(x=280,y=150)
cartButton.config(font=font1) 

losButton = Button(main, text="Predict Length of Stay", command=predict)
losButton.place(x=650,y=150)
losButton.config(font=font1) 

#graphbutton = Button(main, text="CV Error Graph", command=graph)
#graphbutton.place(x=50,y=200)
#graphbutton.config(font=font1) 

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=280,y=200)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
