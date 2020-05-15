# Author : Karima Tajin
# Date : 23 April 2020
# Machine Learning in R 
# Project 4 : Decision Trees and Random Forest 

#############################################
############## Decision Trees ##############
############################################

# Decision Trees are an important type of algorithm for predictive modeling in machine learning 
# Decision tree  utilize a tree structure to model the relation among the features and the potential outcomes. 
# A decision tree classifier uses a structure of branching decisions, which channel examples into a final predicted class value.
# the classical decision tree algorithms have been around for decades and modern variations like Random Forest are among the most powerful techniques available now
# On R, Classification And Regression Trees are known under CART acronym
# Random forests are a strong modeling technique and much more robust than a single decision tree. They aggregate many decision trees to limit overfitting as well as error due to bias and therefore yield useful results.

# in this workbook I will try to predict the houses prices using Decision tree and Random forest

#1- Select the target variable that I want to predict by using glimpse function
#2- Fit a model that can predict the target variable using the predictors variables
#3- Make a few predictions with the predict() function 
#4- plot the decision tree using rpart.plot library

##########################################################################
######### Predict the House Prices using Regression Tree #################
##########################################################################

#get the working directory
setwd('/Users/karimaidrissi/Desktop/DSSA 5201 ML/decision trees')

# loading the required packages
library(tidyverse) # loading tidyverse library
library(rpart) # rpart function implement recursive partition for classification,regression and survival trees
library(rpart.plot) # to plot our data 
library(randomForest) # basic implementation
library(caTools) #for data splitting


#load the dataset
# the dataset is from https://www.kaggle.com/farhankarim1/usa-house-prices
USA_Housing = read.csv("USA_Housing.csv") 
# glimpse the dataset
glimpse(USA_Housing) # there'r 5000 observations and 7 variables 

# renaming the columns
USA_Housing <- USA_Housing %>% rename( AvgIncome= Avg..Area.Income, 
                                       AvgHouseAge=Avg..Area.House.Age,
                                       AvgRooms= Avg..Area.Number.of.Rooms,
                                       AvgBedrooms= Avg..Area.Number.of.Bedrooms)
names(USA_Housing) # there'r 7 features in our dataset 

# set the seed to make the result reproducible
set.seed(123)
# splitting the dataset 
ratio = sample(1:nrow(USA_Housing), size = 0.25*nrow(USA_Housing))
Training = USA_Housing[-ratio,] # 75% of training dataset
Testing = USA_Housing[ratio,] # 25% of testing dataset 

##############################################################################
########################### Making a Regression Tree Model ###################
##############################################################################

# First attempt of building a model
# build a regression tree with the rpart package that can predict the target variable (Price) using the following predictors:
#"AvgIncome"      
#"AvgHouseAge"     
#"AvgRooms"        
#"AvgBedrooms"     
#"Area.Population"
# Assigning a variable named fit_tree to the model using recrsive partitioning and regression trees (rpart)
# Produce a decision tree by training the induction algorithm on the training dataset
fit_tree  <- rpart(Price ~ Area.Population+ AvgHouseAge + AvgRooms+ AvgBedrooms, data = Training)

# plotting an rpart model using prp function 
prp(fit_tree)

# More advanced representation can be produced using rpart.plot
rpart.plot(fit_tree, nn= TRUE)
rpart.control()

# We can interpret the above regression tree as follow:
# the average age of the house and the average area population are important factors for deciding the price of a house 
# the buyer are more likely to buy a house if the average age of a house is less than 6 years with more rooms. 

########################################################################
################### Predicting the price of the houses #################
########################################################################

#Making predictions for 6 houses
head(USA_Housing$Price) # the actual value for the following 6 houses 
# 1,059,033.6 1,505,890.9 1,058,988.0 1,260,616.8 630,943.5 1,068,138.1

# the house price predictions are 
predict(fit_tree,head(USA_Housing))
# 878,186.6 1,443,909.0 1,251,476.5 1,265,810.8 1,089,585.0 878,186.6

# As we can see the actual prices and the predicted values have some differences, maybe using another Machine Learning Algorithm such as Random Forest will help to reduce that difference.

######################################################################
############################### Evaluating the model #################
######################################################################

# Checking the model accuracy either using the Mean Absolute Error(MAE) OR Root Mean Squared Deviation(RMSE)
# Both models can be used to summarize the difference between the actual and the predicted values
#MAE measures the average of the errors in a set of predictions 
#MAE calculate the absolute error between the actual and the predicted 
#Get the mean average error of our model by using mae function
library(modelr)

#Get the mean average error bw the predicted values and the testing dataset
mae(model = fit_tree, data =Testing)
# 225,070.8 the average error in our model 

# using Root Mean Squared Deviation 
predict <- predict(fit_tree, Testing)
# function that returns the Root Mean Squared Error
RMSE <- function(x,y) {
  a <- sqrt(sum((log(x)-log(y))^2)/length(y))
return(a)
}
RMSE1 <- RMSE(predict, Testing$Price)
RMSE1
#[1] 0.2622065 is the average magnitude of the error in our model

#################################################
######### Random Forests ########################
################################################
#Random forest is an ensemble learning method for classification(and regression) that operate by constructing a multitude of decision trees 
# Fitting a random forest algorithm 
set.seed(1000)
fit_random <- randomForest(Price ~ Area.Population+ AvgHouseAge + AvgRooms+ AvgBedrooms, data = Training)
print(fit_random)


# Making predictions for 6 houses
head(USA_Housing$Price) # the actual value for the following 6 houses
# 1,059,033.6 1,505,890.9 1,058,988.0 1,260,616.8 630,943.5 1,068,138.1

# the price predictions for t6 houses using fit_random
predict(fit_random,head(USA_Housing))
#1,007,902.1 1,384,690.8 1,401,091.8 1,254,860.1  832,364.2  751,535.1 

# Random Forest are better in predicting the price for 6 houses 

# summarize the accuracy using Rnadom Forest
#MAE for randomForest model by using testing dataset
mae(fit_random, Testing)
# 202,218 is the mean absolute error value
# using Root Mean Squared Error in random forest model
predict1 <- predict(fit_random, Testing)
RMSE2 <- RMSE(predict1, Testing$Price)
RMSE2
#[1] 0.2353238 is the average magnitude of the error in our random forest model

###########################
#######Conclusion##########
###########################
# the average model prediction error in Random forest is less than Decision tree
# MAE and RMSE both methods fits better in Random Forest compared to Decision Tree Algorithm.



