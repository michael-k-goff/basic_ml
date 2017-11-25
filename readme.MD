# Basic Machine Learning Algorithms

This is a simple project to test machine learning algorithms. We examine four in this project:

* Decision Tree regression
* Support Vector Machine regression
* Linear Regression
* Ensemble

This is a very simple project to test basic concepts and code.

### Data

We test the models on house sale prices for King County, Washington, USA. The data set is available [from Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction). The data gives prices on houses sold from May 2014 to May 2015, and for each house vital data such as ZIP Code, lot size, number of bedrooms and bathrooms, etc.

Our analysis is only conducted on numeric data. Better results are likely when also considering ZIP Code.

### Data Preparation

The full data set consists of 21436 observations, after duplicate observations are removed. The set is divided into a training set (about 60%), dev set (20%), and test set (20%). The training set is used to train the algorithms, the dev set to tune hyperparameters, and the test set to report results. The sets are divided randomly but a seed is chosen to insure they are constant. Experimentation with hyperparameters was done by hand.

### Model construction

The decision tree was developed by tuning the 'min_samples_split' parameter, which is the minimum size of a node on which further splitting is allowed. Through experimentation, it was determined that min_samples_split=50 was the best choice. Smaller values would lead to overfitting (high variance), while larger values would lead to underfitting (high bias).

Experimentation with the SVM regressor hyperparameters failed to produce meaningful results, and so the SVM model was dropped from consideration.

No hyperparameters were used in training the linear regressor.

The ensemble model was constructed by averaging the predictions of the decision tree model and the linear regression model.

### Results

The models are evaluated by their performance on the test set. Performance is assessed by the mean squared error. Here, the errors are reported as a fraction of the total variance of sale prices in the test set.

* Decision tree: 0.349
* Linear regression: 0.326
* Ensemble: 0.284

### Conclusions

This project is only a very test of machine learning concepts, and our models are close to the simplest possible. We do not use ZIP Code, latitude, or longitude data in the analysis, and these features would no doubt improve performance greatly.

### Instructions

To run, download the data set, insure that the needed python packages are installed. Set the project directory path correctly in Line 27 of machine_learning_examples.py.