import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("machine.data", sep=",") #read in all data from database and exclude the commas

#predicting Estimated relative performance
predict = "ERP"

#I chose attributes which are integers and most relivant to improve accuracy
data = data[["MYCT","MMIN", "MMAX","CACH", "CHMIN","CHMAX","ERP"]]
'''
MYCT: machine cycle time in nanoseconds (integer)
MMIN: minimum main memory in kilobytes (integer)
MMAX: maximum main memory in kilobytes (integer)
CACH: cache memory in kilobytes (integer)
CHMIN: minimum channels in units (integer)
CHMAX: maximum channels in units (integer)
PRP: published relative performance (integer)
ERP: estimated relative performance from the original article (integer)
'''

#set up two arrays, one is array defined all of the attributes, and another is the label
x = np.array(data.drop([predict], 1))
y =np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#takes in all of the attributes/labels and split into four different arrays (x_train, x_test, y_train, y_test)
#0.1 so that it splits up 10% of the data into test samples, and let model use 90% information (training dataset) to become good at predicting


#Train model multiple times to achieve high accuracy
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    #linear regression is used as there should be a linear relationship, and the datatypes of attributes in the dataset is integers

    linear.fit(x_train, y_train) #creates the best fit line
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc)) #prints accuracy for each instance from dataset

    if acc > best:
        best = acc
        with open("hardware_performance.pickle", "wb") as f:
            pickle.dump(linear, f)

#Loads the linear model
pickle_in = open("hardware_performance.pickle", "rb")
linear = pickle.load(pickle_in)

#printing coefficient and intercept from the best fit line for all attributes used above
print("------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("------------------------")

# prints a list of all predictions as well as the input data from the dataset
predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

# Drawing and plotting model to find relationships between two attributes
plot = "CACH" #can change to investigate the significance of certain attributes affect on Computer performance(ERP)
plt.scatter(data[plot], data["ERP"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Performance")
plt.show()
