### Overview 
In this practical application, the goal is to compare the performance of the classifiers , namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. A dataset related to marketing bank products over the telephone is utilized for this application.

### Dataset
The dataset is obtained from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. 

### Understanding the Data
The dataset is collected based on 17 marketing campaigns between May 2008 and November 2010 corresponding to a total of 79354 contacts.A subset of this data (cleaned-up) is included in the 'bank-additional-full.csv' file containing a total of 41188 records and 21 attributes. During these phone campaigns, the client was offered an attractive long-term deposit application, with good interest rates. A large number of attributes was stored including personal data, bank information, information related to first and last contact, client's familiarity with the product in the home banking site and information from past campaigns. The target variable 'y' was a binary yes/no variable recorded based on whether the client subscribed a long term deposit. There were 4640 clients who subscribed which corresponded to a  11.3% subscription rate. 

Preliminary exploration of the dataset did not reveal any missing data. There were some categorical variables where the entry is recorded as 'unknown' in several instances. This is not the same as missing data and 'unknown' may be a meaningful category in itself for some attributes. However, in the case of marital status it does not make much sense, hence treating 'unknown' marital status as missing data and dropping those rows seems reasonable- there were only 80 such rows. There were also 12 duplicate entries which were removed. After removal of unknown marital status and duplicate entries, we are left with 41096 records.

### Understanding the Task
The **business objective** is to predict which clients subscribed a term deposit. In terms of machine learning this is a ***binary classification task*** where the goal is to classify clients as subscribers or non-subscribers based on the different attributes related to the bank client data. Some of the attributes related to the campaign like duration of the last contact highly affect the target variable, yet this information would not have been known till the end of the call and hence such atttributes should not be part of a realistic predictor.

### Engineering Features
The seven features corresponding to the bank client data are selected as input features for a classifier model that can predict whether a client is a subscriber or not based on this information. The target variable is converted to numeric by mapping the Yes/No response to 1/0 respectively.We notice that there is substantial imbalance in the data with the subscribers being only 11.2% of the customers. We need to address the imbalance by Synthetic  Minority Oversampling Technique (SMOTE) as many classifiers including logistic regression, KNN, SVM, etc does not train well when there is such imbalance. The oversampling will be applied to training data only, so this will be done after train/test split. We also note that some of the categorical features (e.g. job) have a high degree of cardinality, so we will employ target based encoding for these as one hot or similar form of encoding will greatly enhance the number of features which is not usually desirable for a machine learning problem. We choose Leave One Out encoding which is a form of target encoding that minimizes bias. In this method, the train and test data are encoded separately - the test data is encoded without utilizing the test values of the target variable which is important for not introducing bias in our encoding method. Hence these two steps (SMOTE and Leave One Out Encoding) are performed after the train/test split.

### Train Test Split 
The data is split into train test split using a random_state=42 and default test size of 0.25.

### Categorical Encoding and Scaling
The categorical features are encoded using Leave One Out encoding and the test data is encoded without utilizing the test values of target variable which important for avoiding bias in the test results. After encoding we evaluate correlation matrix to see if there are correlated features in the input data - too many correlated features are best avoided as they lead to multi-collinearity which is not desirable, especially for classifiers like Logistic Regression. There is little to no correlation between the input features. The next step is scaling of features using StandardScaler.

### Balancing Classes in the Training Set
As stated earlier, there is substantial class imbalance in the data and hence the training set is balanced by Synthetic Minority Oversampling technique (SMOTE). The test data however is not oversampled so that the model test results are based on real data (and not on any synthetic data).

### Building and Comparing Simple Models
A baseline model is built using a dummy classifier and different classifiers are evaluated with most of their parameters left to default except for selecting a random_state = 0 (applicable for some classifiers like Decision Tree, Support Vector Machine) and setting max_iter=1000 for Logistic Regression. Following are the results including training accuracy, test accuracy and average fit time for the different classifiers:


   Model	               Train Time	   Train Accuracy	   Test Accuracy
 Dummy	                  0.004997	     0.500000	        0.886899
 Logistic Regression	    0.079321	     0.608528	        0.577380
 KNeighbors	              0.144892	     0.969039	        0.791221
 Decision Tree	          0.081826	     1.000000	        0.837648
 SVM	                  329.568717	     0.650090	        0.604341





