# Import libraries necessary for this project
import numpy as np
import pandas as pd
import visuals as vs

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# 创建‘accuracy_scrore’函数，对预测结果进行打分
def accuracy_score(truth, pred):  
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"  
# Test the 'accuracy_score' function
predictions = np.ones(5, dtype = int)
print(accuracy_score(outcomes[:5], predictions))

# `predictions_0` 函数将始终预测乘客没有存活。
def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(0)
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)
print(accuracy_score(outcomes, predictions))
# 61.62%*
def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """   
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(passenger['Sex'] == 'female')
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print(accuracy_score(outcomes, predictions))
# 78.68%
vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

# 查看存活统计数据后发现，大部分 10 岁以下男性存活了，而大多数 10 岁及以上的男性*没有存活*。
def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(passenger['Sex'] == 'female' or passenger['Age'] < 10)
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)
print(accuracy_score(outcomes, predictions))
#79.35%

vs.survival_stats(data, outcomes, 'SibSp', ["Sex == 'female'"])
vs.survival_stats(data, outcomes, 'Parch', ["Sex == 'female'"])

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """   
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(passenger['Sex'] == 'female' or passenger['Age'] < 10 and passenger['SibSp'] < 3)
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)
print(accuracy_score(outcomes, predictions))
# Predictions have an accuracy of 80.70%.

