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
The seven features corresponding to the bank client data are selected as input features for a classifier model that can predict whether a client is a subscriber or not based on this information.




