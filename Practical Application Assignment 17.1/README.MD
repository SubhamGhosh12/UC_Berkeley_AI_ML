### Overview 
In this practical application, the goal is to compare the performance of the classifiers , namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. A dataset related to marketing bank products over the telephone is utilized for this application. The notebook containing all the codes and results of data analysis and modeling for this application is posted here:

https://github.com/SubhamGhosh12/UC_Berkeley_AI_ML/blob/main/Practical%20Application%20Assignment%2017.1/prompt_IIISG.ipynb

### Dataset
The dataset is obtained from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. 

### Understanding the Data
The dataset is collected based on 17 marketing campaigns between May 2008 and November 2010 corresponding to a total of 79354 contacts.A subset of this data (cleaned-up) is included in the 'bank-additional-full.csv' file containing a total of 41188 records and 21 attributes. During these phone campaigns, the client was offered an attractive long-term deposit application, with good interest rates. A large number of attributes was stored including personal data, bank information, information related to first and last contact, client's familiarity with the product in the home banking site and information from past campaigns. The target variable 'y' was a binary yes/no variable recorded based on whether the client subscribed a long term deposit. There were 4640 clients who subscribed which corresponded to a  11.3% subscription rate. 

Preliminary exploration of the dataset did not reveal any missing data. There were some categorical variables where the entry is recorded as 'unknown' in several instances. This is not the same as missing data and 'unknown' may be a meaningful category in itself for some attributes. However, in the case of marital status it does not make much sense, hence treating 'unknown' marital status as missing data and dropping those rows seems reasonable- there were only 80 such rows. There were also 12 duplicate entries which were removed. After removal of unknown marital status and duplicate entries, we are left with 41096 records.

### Understanding the Task
The **business objective** is to predict which clients subscribed a term deposit. In terms of machine learning this is a ***binary classification task*** where the goal is to classify clients as subscribers or non-subscribers based on the different attributes related to the bank client data. Some of the attributes related to the campaign like duration of the last contact highly affect the target variable, yet this information would not have been known till the end of the call and hence such atttributes should not be part of a realistic predictor.

### Engineering Features
The seven features corresponding to the bank client data are selected as input features for a classifier model that can predict whether a client is a subscriber or not based on this information. The target variable is converted to numeric by mapping the Yes/No response to 1/0 respectively.We notice that there is substantial imbalance in the data with the subscribers being only 11.2% of the customers. We need to address the imbalance by Synthetic  Minority Oversampling Technique (SMOTE) as many classifiers including logistic regression, KNN, SVM, etc does not train well when there is such imbalance. The oversampling will be applied to training data only, so this will be done after train/test split. We also note that some of the categorical features (e.g. job) have a high degree of cardinality, so we will employ target based encoding for these as one hot or similar form of encoding will greatly enhance the number of features which is not usually desirable for a machine learning problem. We choose Leave One Out encoding which is a form of target encoding that minimizes bias. In this method, the train and test data are encoded separately - the test data is encoded without utilizing the test values of the target variable which is important for not introducing bias in our encoding method. Hence these two steps (SMOTE and Leave One Out Encoding) are performed after the train/test split.

#### Exploratory Data Analysis
We also performed exploratory analysis to see how different features individually impact term subscription. Age did not seem to be a differentiating feature for subscription. The categorical features like job, marital status, loan, default, education also seemed to have some influence on the decision to subscribe. From past campaign data, the outcomes from previous campaigns also seem to have a substantial influence. The attributes related to socio-economic context, especially euribor3m, cons.price.idx and cons.conf.idx all seemed to have an influence on term subscription.

### Train Test Split 
The data is split into train test split using a random_state=42 and default test size of 0.25.

### Categorical Encoding and Scaling
The categorical features are encoded using Leave One Out encoding and the test data is encoded without utilizing the test values of target variable which important for avoiding bias in the test results. After encoding we evaluate correlation matrix to see if there are correlated features in the input data - too many correlated features are best avoided as they lead to multi-collinearity which is not desirable, especially for classifiers like Logistic Regression. There is little to no correlation between the input features. The next step is scaling of features using StandardScaler.

### Balancing Classes in the Training Set
As stated earlier, there is substantial class imbalance in the data and hence the training set is balanced by Synthetic Minority Oversampling technique (SMOTE). The test data however is not oversampled so that the model test results are based on real data (and not on any synthetic data).

### Building and Comparing Simple Models
A baseline model is built using a dummy classifier and different classifiers are evaluated with most of their parameters left to default except for selecting a random_state = 0 (applicable for some classifiers like Decision Tree, Support Vector Machine) and setting max_iter=1000 for Logistic Regression. Following are the results including training accuracy, test accuracy and average fit time for the different classifiers:

#### Dummy 
Train Time = 0.004997

Train Accuracy = 0.500000

Test Accuracy = 0.886899

#### Logistic Regression 
Train Time = 0.079321

Train Accuracy = 0.608528

Test Accuracy = 0.577380

#### KNeighbors 
Train Time =  0.144892

Train Accuracy = 0.969039

Test Accuracy = 0.791221

#### Decision Tree 
Train Time =  0.081826

Train Accuracy = 1.000000

Test Accuracy = 0.837648

#### SVM 
Train Time =  329.568717

Train Accuracy = 0.650090

Test Accuracy = 0.604341

All the models have test accuracies that are less than the test accuracy of a dummy classifier. However, given the imbalance in the data, and the need to identify the minority 'subscriber' class, accuracy is not the relevant metric for this application. We want a metric that's sensitive to picking up the small number of subscribers and hence sensitivity (or recall of the minority 'subscriber' class) is a better metric for this application. Let's take a look at how the recall scores compare for these models.

#### Dummy 
Train Recall = 0.00000

Test Recall =  0.00000

#### Logistic Regression
Train Recall = 0.654056

Test Recall =  0.648881

#### KNeighbors
Train Recall = 0.978141

Test Recall =  0.242685

#### Decision Tree
Train Recall = 1.000000

Test Recall =  0.043029

#### SVM
Train Recall = 0.680228

Test Recall =  0.612737

Here the models especially Logistic Regression and SVM seem to be doing a much better job, with a test recall score well above the other classifiers. Also in comparison to the dummy classifier which has a test recall score of 0, the logistic regression classifier has a test recall of 0.65 which looks considerably better.

### Hyperparameter Tuning and GridSearchCV
A user defined function was scripted that can accept the classifier model (clf), train and test data, parameter grid relevant to the classifier and a performance metric to be used as a scoring criterion for GridSearchCV. The function returns the results of GridSearchCV including the grid object with the fitted classifier and the average fit time. It also prints out the classification report for the classifier based on the test data evaluation. In this particular problem we are trying to find the small number of subscribers. It's important for the classifier to maximize the sensitvity to this class because term deposits are important source of income for the bank and the bank, in conducting its marketing campaign, should not miss out on too many customers who could be potential term subscribers. Hence for this problem, the appropriate performance metric is **sensitivity or recall of the minority (subscriber) class**.

After hyperparameter tuning and GridSearchCV using recall as the scoring metric, the Logistic Regression model had the highest recall (0.76) for the test data, followed by SVM (test recall = 0.61) though SVM had a fit time that was several orders of magnitude higher than Logistic Regression. Decision Tree and KNeighbors did not do so well in terms of their test recall score (0.15 and 0.21 respectively) even after hyperparameter tuning. 

### Feature Sub-Selection for Logistic Regression
Since Logistic Regression (LR) was the best performing model based on test recall scores, the relative importance of the input features on the model predictions was explored based on model coefficients. The top four features, in order of importance, were 'job', 'default', 'education', 'marital'. A Logistic Regression model built on these top 4 client attributes and 'hyperparameter optimized' using GridSearchCV yielded a similar test recall score of 0.76 as the Logistic Regression model based on all 7 attributes. The Logistic Regression model with top four client attributes was also the one with the shortest average fit time among all models that were trained and evaluated.

### Other Relevant Performance Metrics
Though recall of the minority 'subscriber' class was deemed as a relevant performance metric, there were other metrics worth looking at for comparing model performances for this application:

**ROC AUC** : ROC curve was generated for all the models. The best ROC-AUC were for SVM (AUC = 0.659) closely followed by Logistic Regression (AUC=0.655 for LR model with top 4 client attributes and AUC=0.654 for LR model with all 7 client attributes). Decision Tree and KNeighbors had ROC-AUC close to 0.5 which implies  no discriminatory advantages for these classification models over a fair coin toss !

**Cumulative Gains Curve Analysis** : This was performed for the Logistic Regression model (with top 4 client attributes) and the SVM model. The cumulative gains curve gives an insight of the potential gain in the percentage of subscribers netted when a certain percentage of clients are sampled based on the model over just random sampling. Both Logistic Regression and SVM had a similar cumulative gains curve. The curve shows that  for a  given sampling percentage (<100%) of clients, the model is going to net more subscribers than just random sampling. For example, using the model to target clients, by sampling just 60% of the clients, it is possible to get about 80% of the term subscribers - a gain of 20% points over randomly sampling 60%.

### Best Performing Model
The best model from the perspective of minimal number of features, shortest average fit time and the best (sensitivity) recall score for the subscriber class is **Logistic Regression classifier which is based on the top four client features, namely job, default, education and marital (in order of importance)**. Logistic Regression with this reduced feature set has similar (or slightly better, in terms of ROC AUC) performance as the Logistic Regression model incorporating all seven features from the bank client data.

Support Vector Machine (SVM)-based classifier is next best which closely matches the performance of Logistic Regression but has the longest fit time, with its average fit time several orders of magnitude higher than Logistic Regression.

KNeighbors (KNN) and Decision Tree classifiers did not work well for this classification problem as reflected by low recall (sensitivity) for the subscriber class and an ROC AUC that's roughly around 0.5 (a random classifier would have an expected ROC AUC of 0.5)

### Adding More Features to Our Best Model

We explored the possibility of boosting the performance of our model by incorporating additional features that seem to influence term subscription based on our initial exploratory analysis. We know that duration of the call should not be included as a feature for a realistic predictor as it is known only after the call has been made. However, there are other information related to month of the call, contact type, outcomes from previous campaigns and prevailing social and economic context attributes which seem to have a substantial impact on decision to subscribe based on our initial exploratory data analysis.Hence, incorporating these features may help improve the performance of the model. This may be useful information to know as well for future campaigns. For example, if there are particular months when customers are more likely to subscribe that's helpful to know. If there are certain prevailing economic/social context attributes that impacts term subscription that's important to know. If the client's response to previous campaigns are known, that may be useful information going forward.  These features may add more valuable information for future campaigns on top of the basic client related data that our models have used this far.

Hence we add the following features to our Logistic Regression model to see if they can enhance the performance over the model that used just the  basic client info.


Information that may affect subscription, e.g. contact type, month of contact, known outcomes from previous campaigns

**contact**: contact communication type (categorical: 'cellular','telephone')

**month**: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

**poutcome**: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')



Information on prevailing social/economic conditions: these are not related to the client but may affect client's subscription

**cons.price.idx**: consumer price index - monthly indicator (numeric)

**cons.conf.idx**: consumer confidence index - monthly indicator (numeric)

**euribor3m**: euribor 3 month rate - daily indicator (numeric)


It is to be noted that emp.var.rate and nr.employed have strong correlation wit euribor3m, so only one of these, euribor3m, is retained for the model as strongly correlated features would lead to multi-collinearity which is not ideal for Logistic Regression.

#### Results of Logistic Regression Model with Added Features
The above features are added on top of the four best client-related features to Logistic Regression Model. After performing GridSearchCV the following model turned out to be the best estimator with the added feature set: 
 
            LogisticRegression(C=0.001, max_iter=1000, penalty='l1', random_state=0, solver='liblinear') 
 
The recall score for the minority 'subscriber' class (sensitivity) dropped a little bit from 0.76 to 0.74 but the recall score for the majority 'non-subscriber' class increased substantially from 0.45 to 0.72. In the end this may be a better model from the perspective of ROC curve area and Cumulative Gains analysis.

The ROC analysis showed an AUC of 0.79 for the Logistic Regression Model with the added features, which is a substantial improvement over the AUC of 0.655 for the LR model with the top 4 client-data based features. There's also an improvement evident from the cumulative gains curve:  over 60% of subscribers can be reached by sampling only 20% of the  clients based on this model. This is a 40% points gain over just randomly sampling 20% of the population which would have expectedly netted only 20% of subscribers. To reach 80% of subscribers, just 50% of the clients need to be sampled based on the model which reflects a gain in subscribers by 30% points compared to just randomly sampling 50%.

#### Recommendations and Future Steps
A Logistic Regression Model built with top four client-data based features had a recall score of 0.76 and ROC-AUC of 0.655 with cumulative gains curve showing important gains in netting subscribers from sampling the clients based on the model compared to random sampling. The model however can be improved substantially in terms of ROC-AUC and cumulative gains if additional features related to type of client contact (telephone or cellular), month of contact, outcomes from previous campaigns and socio-economic attributes like euribor3m (3 month interest rate), consumer price index and consumer confidence index are added to the model on top of the four client-data based features. 

Cumulative gains curve for the model showed that using the model to target a certain percentage of the clients can lead to substantial gain in the number of term subscribers compared to randomly selecting clients for the campaign. For example, using the model to select clients, over 60% of subscribers can be reached by sampling only 20% of the total number of clients. To reach 80% of subscribers, just 50% of clients need to be sampled if we are using the model. This clearly shows the value of using the model for a campaign as there are typically time and budget constraints which may limit the marketing team from calling up the entire database of customers. Intelligent targeting of a smaller percentage (e.g. 20%, 50%, etc) of customers using the model will enable the bank to reach more term subscribers compared to random selection of the same percentage of clients from the database.

Potential next steps may involve evaluating other classifiers like Naive Bayes, Random Forest and Gradient Boosted Trees to see if they can provide a meaningful improvement in classification.
 




