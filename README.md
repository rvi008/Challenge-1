# Report of Machine Learning course's challenge

## First Step : Data Analysis
<p align="justify">
First, we plot, with seaborn, the correlation matrix of the features. We can see that there isn't any visible correlation between our different features.
We can also see that each of the features follows a normal law and there isn't any visible "flaw" in the data, meaning that there's not so much feature engineering to do in order to fill missing values, filter outliers etc ...
Nevertheless, we do a feature selection using a LogisticRegression estimator and the Recursive Feature Elimination Class.
With this method, we can remove 9 features. 
</p>
<figure>
<img src="challenge/dist.png" alt="Plots of features' distribution>
<figcaption align="center">Plots of features' distribution</figcaption>
</figure>
<figure>
<img src="challenge/corr.png" alt="Correlation matrix of the variables">
<figcaption align="center">Correlation Matrix of the training variables</figcaption>
</figure>

## Second Step Model Selection
### Neural Network
<p align="justify">
We start with a "reference" model which is the one provided in the starter notebook e.g a Logistic Regression.
The score is not so good because we end with a ~0.6385 score on the training set which is very high and means our classifier is under-fitting the training set.
Anyway, we obtain a score of ~1.381 which is worst than predicting 0 for each of the observations e.g undefined  
As the improvement in score is not significant without the removed columns, we chose to take to reinclude them into 
the training dataset.
We are told the dataset are variables build from pictures and we know Neural networks are efficient for this type of 
problem. We start with a "base" Multi-layer perceptron from Scikit learn e.g using default values from the python's class.
The fitting score on the training data-set is nearly perfect : ~0.00028 and on the leaderboard we get a ~0.34 score.
Now, to improve the score we check the effect of Standardization over the training dataset. We use the StandardScaler from SKLEARN
and predict over the training dataset. The score decreased, and it's perhaps linked to the fact that the variables are all Standardized
gaussian distribution. 
Anyway, we start with this "base" model and the non-standardized dataset and try to optimize the hyper-parameters.
</p>
In order to do that we use the GridSearchCV class and cross-validate the following parameters : 
* max_iter [100, 300, 500, 1000]
* hidden_layer_sizes [2:20]
* activation ['tanh','identity','logistic','relu']
* alpha [0.0001:0.001]

This GridSearch vas computationnally intensive and took about a night. Finally, we get the following parameters :
* max_iter : 300
* hidden_layer_sizes : 14
* alpha : 0.0002

<p align="justify">
The training score for this model is ~0.14 which significantely worse than the default MLP one's. And without any suprise the score
on the leader bord is not better than the previous one.
We now explore an other way to improve the score on the leaderboard : don't predict any class when we are below or above a treshold
for the probability of the observation to belong to a class.
We can see by plotting the distributions of probabilities of the 2 classes that there are tails near 0% and 100% of probabilities to belong to the class.
So we set the prediction to 0 for the observations which are between two bounds found empirically. The best score obtained this way is
~0.20.
</p>

<p align="justify">
To improve our score we'll proceed differently, we split our train set in learning / validation set with 50/50 splitting. Then we will do 
several grid search to identify the best parameters we can. We start with broad ranges of values (e.g logspaces) for hidden layer sizes,
learning rates and we also search the best solver / activation function. Quickly we see a significant impact of the relu function and the
adam solver. For the two other parameters, we narrow down our research using linspaces, with three other Grid Search CV  we get : </p> *hidden layer
sizes = (168,) eg one layer of 168 neurons and *alpha = 0.01444.<p align="justify"> Thereafter, we predict on the validation data left aside with this model
trained on half the trainning set. The validation score is ~0.186 which is promising especially if we don't trust the predictions which 
probabilities are under 90% for positive classification or above 10% for negative classification. Finally we predict on test data and get
a ~0.1565 score.
</p>   


### K-nearest neighbours
<p align ="justify">
We start with a simple KNN classifier : weights set to 'distance', p = 1, algorithm='kd_tree'.The fitting score is perfect (0.0 error) and suggest that we are overfitting. Anyway
we get a ~0.5 score on the leaderboard which is quite disapointing compared to the first score we got with the Neural Network.   
We try a grid search on the number of neighbors and the p parameter setting the distance.
The result given by the Cross validated grid search is : K = 6 neighbors, weights = distance and leaf size = 25, we then fit our model on the 
whole trainning set. The result we got is worst than the MLP's one.




