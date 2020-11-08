# Supervised-Machine-Learning-Model-to-predict-Computer-Performance
Brief description: Model uses Linear Regression to predict computer performance using independent hardware attributes from UCI repository dataset. Learned and used the Python libraries ‘Pandas’ and ‘Numpy’ to read key attributes from dataset. Additionally ‘SKLearn’ and ‘Pickle’, to train, and load the model. Enabled visualisation of relationship between an attribute in the dataset and the relative performance using ‘matplotlib’ library.

Opening HardwareML.py
It might take a couple of seconds for the program to run once clicked. The attribute variable against relative computer performance for plotting the scattar diagram is currently Cache memory (CACH).


Description of program
Supervised Machine learing Linear Regression model to predict computer performance from independent computer hardware attributes/variables.
linear regression is an algorithm finds a line (y=mx+b) that best fit for scatter of data points on the plot.
If there is a fairly strong correlation, the line can be used to predicts ouputs from given inputs.
This model is supervised as the algorithm learns on a labeled dataset, providing an answer key that the algorithm can use to evaluate its accuracy on training data. 
Can change independent attributes/variables to investigate the significance of certain attributes affect on Computer performance(ERP)


What I have learned
- most significant attributes which positively affected computer performance were the increase of MMIN (minimum main memory) and MMAX (maximum main memory)
- How linear regression is used computationally
- Libraries: pandas, sklearn, pickle, matplotlib and how each of them played a pivital role in gathering, training, loading and plotting data from the dataset.
  (More specific) Used the Python libraries ‘Pandas’ and ‘Numpy’ to read key attributes from dataset, ‘SKLearn’ and ‘Pickle’ to train & load the model. 
  Enabled visualisation of relationship between an attribute in the dataset and the relative performance using matplotlib.


How I could improve
- Boost the accuracy of the model
	-use datasets with more instances to get more accurate the results
	-treat missing, and detecting and removing outliers
	-use more algorithms that use linear regression to find and use the one with the highest percentage of accuracy
- Create a GUI, take in user inputs and uses the model to return output


Ideas for the future
- Try to make a model that uses classification or clustering 
- store data based on program analysing drawn pictures e.g. to predict if people have parkinsons or dementia **
- gather data yourself from an experiment and use them on the model
- Try make a unsupervised  model


Resources used to help
- 'Tech with Tim Machine Learning Tutorial [#1-4]' on Youtube which is also on his website. 
  Youtube:https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg
  Website: https://techwithtim.net/tutorials/machine-learning-python/introduction/
  Tim's tutorial focused on using a student performance database to predict student's final grades. 
  To seperate myself and further understand the innerworking of the model, I used the Computer Hardware database (from UCI repository database),
  Additionally I am interested to investigate the relationship between certain hardware attributes to the relative computer performance. 
- The dataset I used is from UCI Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
  Accessed in 2020
  Creator:
  Phillip Ein-Dor and Jacob Feldmesser
  Ein-Dor: Faculty of Management
  Tel Aviv University; Ramat-Aviv;
  Tel Aviv, 69978; Israel

  I made sure the database I used from UCI repository dataset had a lot of instances to increase the accuracy of the model in prediction.



