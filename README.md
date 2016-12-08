# Report of Machine Learning course's challenge

## First Step : Data Analysis
<p align="justify">
First, we plot, with seaborn, the correlation matrix of the features. We can see that there isn't any visible correlation between our different features.
We can also see that each of the features follows a normal law and there isn't any visible "flaw" in the data, meaning that there's not so much feature engineering to do in order to fill missing values, filter outliers etc ...
Nevertheless, we do a feature selection using a LogisticRegression estimator and the Recursive Feature Elimination Class.
With this method, we can remove 9 features. 
</p>
<figure>
<img src="challenge/corr.png" alt="Correlation matrix of the variables">

<figcaption align="center">Correlation Matrix of the training variables</figcaption>
</figure>


## Second Step Model Selection
<p align="justify">
We start with a "reference" model which is the one provided in the starter notebook e.g a Logistic Regression.
The score is not so good because we end with a *~0.6385* score on the training set which is very high and means our classifier is **under-fitting** the training set.
Anyway, we obtain a score of *~1.381* which is worst than predicting 0 for each of the observations e.g undefined  
