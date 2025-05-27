import matplotlib.pyplot as plot
import pandas
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import *
from sklearn.model_selection import *

weatherData = pandas.read_csv("DQN1Dataset.csv")

print(weatherData.shape)
#drops rows that contain duplicate values of the datetimeEpoch column
weatherData = weatherData.drop_duplicates(subset =["datetimeEpoch"])
#drops rows with missing values, in this case nan because pandas reads empty csv boxes as nan
weatherData = weatherData.dropna()
print(weatherData.shape)

#prints all unique values within csv dataset
print(weatherData.nunique())

# plot.hist(weatherData["temp"], bins=10)
#
# plot.title("histogram")
# plot.xlabel("")
# plot.ylabel("")
#
# plot.show()

# plot.boxplot(weatherData["healthRiskScore"])
# plot.show()

features = ["pm2.5", "no2", "co2", "temp", "humidity", "windspeed"]

#loads train/test variables using sklearn lib. Weather conditions/quality as input and healthRiskScore as target
#sets hyperparameters to achieve the more accurate results
trainX, testX, trainY, testY = train_test_split(weatherData[features], weatherData["healthRiskScore"], test_size=0.8, random_state=20)

#random_state(seed), creates sklearn randomForest object
randomForest = RandomForestRegressor(random_state=5)

#creates new gridSearch model that will try the hyperparameterGrid values to determine most the most effective
#optimizes randomForest to achieve the most accuracy
hyperparameterGrid = {"n_estimators": [25, 70],"max_depth": [5, 20]}
gridSearch = GridSearchCV(estimator=randomForest, param_grid=hyperparameterGrid, )


#trains gridSearch then saves the best model
gridSearch.fit(trainX, trainY)
gridSearch = gridSearch.best_estimator_

#after fitting model the best estimator is loaded then used to determine the importance of included features
#features below threshold of importance are removed to implement feature importance
threshold = 0.09
i = 0
for importance in gridSearch.feature_importances_:
    if(importance <= threshold):
       features.pop(i)
    ++i

#updates test/train variables after removing unimportant columns
trainX, testX, trainY, testY = train_test_split(weatherData[features], weatherData["healthRiskScore"], test_size=0.8, random_state=20)

#implements gradientBoosting and stacking ensemble methods to increase the overall accuracy of the model
modelStack = [('rf', gridSearch),('gb', GradientBoostingRegressor(n_estimators=100, random_state=5))]
stackModel = StackingRegressor(estimators=modelStack, final_estimator=LinearRegression())
stackModel.fit(trainX, trainY)

#saves predicted values using trained stackModel object and using random forest with gradient boosting in stack
predictionY = stackModel.predict(testX)

#printing of model MSE/R^2 scores along with visualized results on scatter plot
#smaller value indicates better prediction accuracy
print("\nMSE score: " + str(mean_squared_error(testY, predictionY)))

#0-->1 means good fit, R^2 score:
print("\nR^2 score: " + str(r2_score(testY, predictionY)))

#avg difference predicted/actual values
print("\nMAE score: " + str(mean_absolute_error(testY, predictionY)))

#R^2 but tolerates variance more
print("\nExplained Variance: " + str(explained_variance_score(testY, predictionY)))

plot.scatter(testY, predictionY)
plot.xlabel("real value")
plot.ylabel("prediction")
plot.show()

# assessment scores -task1
# MSE score: 0.08850124247520247
# R^2 score: 0.8098212289121004

#After optimization (feature importance, hyperparameter tuning, stacking)

# assessment scores -task2
# MSE score: 0.0715407788331126
# R^2 score: 0.8462672724062065
# MAE score: 0.197082605986312
# Explained Variance: 0.8465798205408627
