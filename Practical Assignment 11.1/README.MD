# **Practical Application Assignment 11.1 (Module 11, UC Berkeley Professional Certification in AI/ML):** 
# ***What Drives the Price of a Car?***


This file contains a summary of findings based on exploration of a Kaggle dataset ( recorded in the csv file "vehicles.csv") on used car prices. The CRISP-DM framework is followed as closely as possible for this work as described below. The codes for data cleaning and visualization, modeling and other analyses are found here:
https://github.com/SubhamGhosh12/UC_Berkeley_AI_ML/blob/main/Practical%20Assignment%2011.1/PromptIIa.ipynb

## **Business Understanding** 

The key objective  from the business perspective is  identifying which  attributes impact the price of used cars. This information will be valuable to used car dealers in fine tuning their inventory. This can be achieved building a  regression model for price of used cars based on different attributes including year, make, model, condition, odometer reading, number of cylinders, type of fuel, drive, transmission, size, type of vehicle and color. The magnitudes of regression coefficients as well as other model inspection techniques can reveal useful information in  ranking features in order of their impact on the target variable ('price') and accordingly determine the top features that influence price.

## **Data Understanding**

The first step of data understanding is exploring what data has been collected. In this case there are 426880 entries with 18 attributes (id, region, price, year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, VIN, drive, size, type, paint color, state):  4 numeric attributes (id, price, year, odometer) and the rest categorical.


 A quick check of the descriptive statistics of the numeric attributes indicate that there are outliers in numeric attributes, price and odometer. Based on vehicles.info() there is a large number of missing data from the following columns: condition, cylinders, VIN, drive, size, type, paint color. These features will therefore not be used for the model as there's no reliable way to replace so many instances of missing data corresponding to these features without introducing some sort of error or bias. There are other features like year, manufacturer, model, fuel, transmission, state, odometer reading, each of which is known to impact used car price. There's a smaller amount of missing data for these features and there will still be sufficient data left after elimination of missing data corresponding to these features. The price and odometer columns seem to have some outliers based on the descriptive statistics: for example the max value of price seems unrealistically high whereas minimum value is 0. Contents of categorical features like fuel, transmission are also explored.
 
 ## **Data Preparation**
 
 First step in data preparation process is removal of outlier data for the numerical columns 'price' and 'odometer'. This is accomplished based on removing data more or less than 1.5 times the interquartile ranges from the third and first quartile respectively. Also since several datapoints for the price variable are on the lower side (0 or close to 0), the lower limit may not be adequate for removal of outliers on the lower side of the range, so a value threshold of 500 is set below which data for price is removed. There are a large number of missing data from the columns condition, cylinders, VIN, drive, size, type, paint color. These features will therefore not be used for the model. The 'region' and 'state' both convey information on location, it will be adequate to include just one of them, 'state', for modeling purposes as variation in price for regions within a state are probably going to be not as much as variation from one state to another. Before getting the data ready for modeling we explore the effect of a few variables on price to help us make important decisions on what to incorporate in the model. This is easier to do at this stage after the outlier prices have been removed as one of the first steps of data preparation.

Based on the above analysis and availability of data, we choose the following features for our model:  year, manufacturer, model, fuel, odometer, title_status, transmission, state as input features and price as the obvious target variable. These are extracted into a new dataframe, NaN values are dropped from these columns and we are left with a dataframe that has 338533 rows and 9 columns.

The ranges of numeric data appear reasonable even though there's still a difference in scales especially when comparing amongst the different features. For example, year is in the scale of thousands, price can be of the order of tens of thousands while odometer can go up to the order of hundreds of thousands. It is clear that ***scaling*** will be important when we develop models and we plan to incorporate the StandardScaler in our pipeline for the different models to take care of differences in scales.

There are several categorical variables like manufacturer and model which are believed to have major impact on price. These are not ordinal. Due to the high cardinality of these categorical features, one-hot encoding is not used. Instead, leave one out encoding is used which will avoid introducing additional columns/features for each of these categorical variables while at the same time minimizing the likelihood of overfitting. Leave one out encoding is just target encoding where the average or expected value is calculated ignoring the value in the current row. It is believed to have lesser issues with bias and overfitting than traditional target encoding, hence it is deemed to be a good choice for this particular problem. Category encoders is installed so that LeaveOneOutEncoder can be used.

Next step is dividing data into train test splits for ***hold out cross-validaton*** using the default test size of 0.25 and setting the random_state to a fixed integer so that the same train/test sets are used for multiple calls comparing across different models. After train test split, the categorical variables are encoded by leave one out encoding. The approach to leave one encoding is different for the test set compared to the train set. Since leave one out encoding utilizes the value of the target variable ('price' in this case), the encoding for the train set is done first and then the test set is encoded without utilizing the test values of the target variable. **This is important so as not to overfit and/or bias the results (due to'data leakage') of the model on the test dataset.** 

## **Modeling**

The modeling stage involves building a variety of models starting with linear regression  and then going to the ridge and lasso models. Hold-out cross validation is performed to assess the performance of each model on the test data.

#### **Simple First Order Regression Models**

We start with building simple first order models first without introducing polynomial complexity. Though simpler models may be less accurate they are easier to interpret in terms of the regression coefficients and the relative impact of different features on target variable. Mean squared error (MSE), Root Mean Squared Error (RMSE) and r2 scores were used for quantifying model performance. Below is a summary of the performance of different models (simple first order models):


**Linear Regression Model: Model Scores** 

Train MSE: 41136000.72

Train r2 score: 0.74

Test MSE: 41987760.46

Test RMSE: 6479.8

Test r2 score: 0.73

**Ridge Regression Model: Best Alpha and Model Scores**

Best Alpha(based on **GridSearchCV**): 10.0

Train MSE: 41136000.89

Train r2 score: 0.74

Test MSE: 41987805.35

Test RMSE: 6479.8

Test r2 score: 0.73

**Lasso Model: Best Alpha and Model Scores**


Best Alpha (based on **GridSearchCV**): 0.001

Train MSE: 41136000.72

Train r2 score: 0.74

Test MSE: 41987760.48

Test RMSE: 6479.8

Test r2 score: 0.73

Regularization (Ridge and Lasso) did not help improve the performance over simple linear regression. Based on examining the regression coefficients all three models consistently had the following as the top five features (ranked in order of magnitude of regression coefficients): car model, odometer, year, transmission and fuel. 

***Sequential Feature Selection (SFS)*** with number of features =5 also identified the same five features as listed above. SFS with simple first order linear regression model had the following scores:

Train MSE: 42529394.36

Train r2 score: 0.73

Test MSE: 43472325.81

Test RMSE: 6593.35

Test r2 score: 0.72

Sequential Feature Selection (SFS) showed how a simpler model can built with fewer (5) features, where the top features in order of magnitude of coefficients, and hence in the order of highest (positive or negative) impact on price are:

1) Model (positive)

2) Odomoter (negative)

3) Year (positive)

4) Transmission (negative)

5) Fuel (positive)

This is consistent with what the earlier models (without SFS) including  Linear Regression, Ridge Regression and Lasso models showed based on their respective model coefficients.




#### **Higher Order Polynomial Regression**

Next we evaluate the effect of increasing complexity with **Polynomial Features** and iterating through different complexites from degree = 1 through degree = 4 and a **LinearRegession estimator**. We evaluate the best complexity based on the test mean squared error and accordingly found the best complexity as degree = 2, with the following scores:

Train MSE :  34174552.49

Train r2 score :  0.78

Test MSE :  34768913.41

Test RMSE : 5896.52

Test r2 score for Best Model :  0.78

The second order polynomial regression with the Linear Regression estimator seems to give a meaningful improvement in model performance than simple first oder linear regression. We also tried the Ridge and Lasso models with second order polynomials and following are the model scores:

**Ridge Regression with Higher Order Polynomial (degree=2)** : Best Alpha and Model Scores

Best Alpha (based on **GridSearchCV**): 0.001

Train MSE: 34174553.86

Train r2 score: 0.78

Test MSE: 34769171.24

Test RMSE: 5896.54

Test r2 score: 0.78

**Lasso Model with 2nd order Polynomial Features**: Best Alpha and Model Scores


Best Alpha (based on **GridSearchCV**): 1.0


Train MSE: 38207964.61

Train r2 score: 0.76

Test MSE: 38967647.78

Test RMSE: 6242.41

Test r2 score: 0.75

The best performing model based on these evaluations was the Linear Regression model with second order polynomial features. Since this was a complex model, interpretation of coefficients was not that simple. Hence **permutation importance** analysis was performed to examine which features had most impact on model results. These are the features ranked in order of their importance (highest impact first):

neg_mean_squared_error

    model   :   91342721.779 +/- 459620.918
    
    year    :   26984415.548 +/- 239674.192
    
    odometer:   24307979.473 +/- 243604.737
    
    transmission:   6785544.901 +/- 86889.602
    
    fuel    :   6735853.617 +/- 81313.039
    
    manufacturer:   1840768.305 +/- 52190.469
    
    title_status:   1725386.463 +/- 43361.966
    
    state   :   960289.676 +/- 34660.013

We built a second order polynomial model with the top five features based on the above results from permutation importance analysis. There was a slight loss of performance, so overall our best model was the Linear Regression Model with 2nd order Polynomial Features based on eight attributes. However the top five attributes driving used car prices (as we saw consistently across different models) were car models, year, odometer, transmission and fuel.

## **Model Evaluation**

Our best performing model is second order polynomial model with Linear Regression that accepts 8 attributes as input features to predict the price of used cars. We have used hold out cross-validation for our model which works well for most cases. In the evaluation phase, we re-tested the performance of the model using K-Fold validaton just to make sure our results from the hold out cross-validation are not biased due to any unprecedented reasons. Once again we made sure test datasets are encoded without utilization of the test values of the target variable to make sure we get unbiased results from K-Fold Cross-Validation. Following are the results from K-Fold validation using number of splits=5.

KFold Scores for number of splits = 5

Test MSEs: 
 [34563702.2, 34875873.75, 35541773.18, 35088180.86, 34511662.45]

Mean Test MSE: 34916238.49

Test RMSEs: 
 [5879.09, 5905.58, 5961.69, 5923.53, 5874.66]

Mean Test RMSE: 5908.91

Test r2 scores:
 [0.78, 0.78, 0.77, 0.78, 0.78]

Mean Test r2 score: 0.78

The results from K-Fold cross-validation are consistent with the results obtained earlier from hold out cross-validation. **This indicates that the model performance is reproducible across different train-test sets and hence it can be expected that the model will have a similar performance with unseen data.** 

We studied the distribution of the training and test residuals ('prediction errors') for our best model and both were normally distributed with a peak around zero. Based on examining the relationship of prediction error vs the target variable price, the model seemed to have more errors at the low or high end of prices and it worked better in the mid range.

*Summary of Model Evaluation and Next Steps*

The second order polynomial model with Linear Regression turned out to the best performing regression model for prices of used cars based on the set of input features including manufacturer, model, year, odometer, fuel, transmission, title status, state. The model is complex but is more accurate than other models we evaluated, with a mean squared error of  34768913.41 (RMSE = 5896.52) on the hold out test set and similar performance metrics with KFold cross-validation. The model is able to account for more than 75% of the variance in the data (based on a test r2 score of 0.78) which is expected from a high quality model. Mapping back to the input feature set, the top five features with most impact on prices are as follows (highest impact first)

1) model

2) year

3) odometer

4) transmission

5) fuel

The manufacturer did not figure among the top features driving prices. Interestingly, this top five feature set with highest impact on prices is consistent with the top five derived from simpler regression models based on magnitudes of the corresponding regression coefficients or using Sequential Feature Selection to identify top five features for a simple linear regression model. However, the order in terms of descending magnitude of coefficients is slightly different. For the simpler regression models, the car model was still the one with the highest coefficient (suggesting highest impact), then came odometer, year, transmission and fuel in that order (so the only change in order was odometer and year switching their places up and down the order respectively).

Potential next steps may include evaluating a model with more advanced regressors (e.g. Random Forest) on this used car dataset and assess any potential boost in performance. Even though the predictive performance of the model may be improved, it is unlikely that the main conclusion on top attributes driving prices will change majorly, given the consistency we saw with different regression models that were evaluated in this project.

## **Deployment**

A brief report (listed below) is created as a potential deliverable to the client. 

**Objective**:  The goal of this analysis was *identifying the attributes that had the highest impact on used car prices.*  

**Data**: A dataset on used cars from Kaggle was explored for this purpose. After cleaning the data and retaining the attributes deemed to be relevant for modeling, the final dataset had  338533 entries with 8 attributes including manufacturer, model, year, odometer, transmission, fuel, title status and state,  and 1 target variable, price.  

**Modeling**: Different regression-based machine learning models were explored for modeling the price of used cars using 8 different attributes as mentioned above. We started with simpler models which may not be the most accurate but are easier to interpret and then proceeded to develop more complex models with higher degree polynomial features. Each of the models were explored in terms of the regression coefficients or permutation feature importance analysis in assessing the relative importance of features from the perspective of their influence on the predicted price.

**Results**: The best performing model was a Linear Regression model with second order polynomial features.  Based on model inspection the following five features came out on top in terms of their impact on price (highest impact listed first):

**1) model**

**2) year**

**3) odometer**

**4) transmission**

**5) fuel**

Manufacturer was not among the top five features impacting price. Interestingly, these top five features were consistently regarded as most impactful from other regression models that were explored.


**Recommendations**: **Models** of car have the biggest impact on prices. In terms of inventory, this would imply stocking up more on premium models across different manufacturers for favorable pricing rather than have different tiers of models corresponding to  a particular manufacturer. Stocking the inventory with more recent models is also important as **year** is the attribute with second most impact on price. Cars that have been driven less fetch higher prices as **odometer** is the third most impactful feature (lower the odometer, higher the price). Lastly, **transmission** (automatic or others for higher price compared to manual) and **fuel** type (diesel or electric for higher price compared to gas/hybrid) also impact price but to a lesser extent than the other three.

 
 
