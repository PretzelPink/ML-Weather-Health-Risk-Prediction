Project Summary
----------------
Students are provided a csv file with various weather data and will be tasked with predicting the likely health risk depending on weather conditions. The student will assess various machine learning models and implement the model that best suits the needs of the problem. The student will then apply various optimization techniques to enhance the 
model accuracy and speed.

I decided to use the Random Forest machine learning model because of its resistance to noise along with feature importance, data processing, hyper parameter tuning, and stacking. The resulting project removes duplicates/rows with missing data from the csv data file before training a random forest model on the data. Afterwards
the most important features are found and saved along with hyperparameters are tuned using grid search which automatically selects the best combinations before finally creating a final stack consisting of the random forest and a new gradient boosted model that adds a small precision boost.

The model is tested using R^2 and MSE test metrics and displays an accuracy of >80%


-----------------
Skills: Machine Learning, Data Analysis, Data Processing, Machine Learning Optimization, Pandas, Sklearn, Matplotlib
-----------------
