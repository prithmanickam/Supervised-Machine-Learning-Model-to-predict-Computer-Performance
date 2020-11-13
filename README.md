# Supervised-Machine-Learning-Model-to-predict-Computer-Performance

Opening HardwareML.py
It might take quite a number of seconds for the program to run once clicked. The attribute/variable against relative computer performance for plotting the scattar diagram is currently Cache memory (CACH).

Description of program
This program is Supervised Machine learing Linear Regression model to predict computer performance from independent computer hardware attributes/variables. The program also plots a scatter graph of a chosen attribute inrelation to relative computer performance, and by altering the variable it can be found which attribute affects the performance more or less (to give users a understanding of which stats of a CPU they should focus on upon purchasing one).
This model is supervised as the algorithm learns on a labeled dataset. The type of supervised learning is regression as the data I am using are only integers. Linear regression is an algorithm finds a line (y=mx+c) that best fits for scatter of data points on the plot. If there is a fairly strong correlation, the line can be used to predicts ouputs from given inputs.
I split the dataset into test and training data set, where 90 percent is used for the training dataset and 10% is used for the test dataset. A high percentage is used for the training dataset to increase accuracy of the prediction.

What I have learned
- Most significant attributes which positively affected computer performance were MMAX (maximum main memory)
- How linear regression is used computationally
- Libraries: pandas, sklearn, pickle, matplotlib and how each of them played a pivital role in gathering, training, loading and plotting data from the dataset.
  (More specific) Used the Python libraries ‘Pandas’ and ‘Numpy’ to read key attributes from dataset, ‘SKLearn’ and ‘Pickle’ to train & load the model. 
  Enabled visualisation of relationship between an attribute in the dataset and the relative performance using matplotlib.

How I could improve
- Boost the accuracy of the model
	-use datasets with more instances to get more accurate the results
	-treat missing, and detecting and removing outliers
	-use more algorithms that use linear regression to find and use the one with the highest percentage of accuracy
- Create a more genreal purpose ML model with a GUI for users to investigate and uncover hidden insights from their own data unrelated to computer performance.

Ideas for the future
- Supervised model that uses classification or clustering 
- Store data based on program analysing drawn pictures e.g. to predict if people have parkinsons or dementia
- Gather data yourself from an experiment and use them on the model
- Unsupervised ML model

Resources used to help
- 'Tech with Tim Machine Learning Tutorial [#1-4]' on Youtube and also on his website. 
  Youtube:https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg
  Website: https://techwithtim.net/tutorials/machine-learning-python/introduction/
  Tim's tutorial focused on using a student performance database to predict student's final grades. 
  To seperate myself and further understand the innerworking of the model, I used the Computer Hardware database (from UCI repository database).
  Additionally I am interested to investigate the relationship between certain hardware attributes to the relative computer performance. 
- The dataset I used is from UCI Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
  Accessed in 2020
  Creator:
  Phillip Ein-Dor and Jacob Feldmesser
  Ein-Dor: Faculty of Management
  Tel Aviv University; Ramat-Aviv;
  Tel Aviv, 69978; Israel
