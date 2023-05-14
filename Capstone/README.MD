### Project Title : Machine Learning Approach to Identify Risk Factors of Mortality in US Population

**Author** : Subham Ghosh

#### Executive summary

#### Rationale
 Understanding and identifying risk-factors and sub-groups of the population that are at increased risk for mortality is important for increasing average life expectancy, a key metric in human development index. It is also important from the perspective of instituting socio-economic and/or healthcare policies for population at risk,  targeting better managed healthcare as well as social support for these groups to potentially extend their lives. Large national studies of mortality are relatively rare. Based on some of the studies that have been done,  mortality has been linked to different socio-economic factors as well as factors related to gender, ethnicity, race, geography, availability and access to quality healthcare. However, the relative importance of these different factors on mortality is not well understood.
 
#### Research Question
The key objective of this project is to build machine learning models for predicting mortality and identifying the top risk factors associated with mortality in a nation-wide sample of US population.

#### Data Sources
The National Longitudinal Mortality Study (NLMS) is a national, longitudinal, mortality study sponsored by the National Heart, Lung, and Blood Institute, the National Cancer Institute, the Health Statistics part of the Center for Disease Control and Prevention and the US Census Bureau for the purpose of studying the effect of differentials in demographic and socio-economic characteristics on mortality. This data has been provided  upon request submitted at https://biolincc.nhlbi.nih.gov/studies/nlms/

#### Methodology

The CRISP-DM framework is followed as closely as possible for this project. 

***Business Understanding***

The objective is to identify key demographic, geographic and socio-economic risk factors for mortality in US, so that future policies can be instituted that can potentially benefit the population at risk in terms of mortality prevention and increasing average life span.  

***Data Understanding***

The data includes a series of cross-sectional, national samples of the U.S. population, as identified from the Annual Social and Economic Supplement (ASEC – formerly the Current Population Survey) of the Bureau of the Census. The latest release of NLMS, version 5,  includes data files for public use from a survey with full follow-up for 11 years (n=1,835,072). Records from the Annual Social and Economic Supplement (ASEC), beginning in March 1979 were matched to the National Death Index to identify the occurrence and cause of death through 2011. Extensive demographic, social, economic, and occupation information is collected in the ASEC. The linkage of the individual social and economic data with the mortality outcomes provides the resource for extensive analysis. The data corresponding to categorical attributes are pre-encoded using numeric values so that all columns have numeric entries. A full explanation of the different attributes and encoding schemes for categorical features can be found in the Reference Manual that came with this dataset.

***Data Preparation***

The data is cleaned following standard procedures including removal of columns with too many missing entries (>1000000), and attributes like record id, cause of mortality, houshold number, length of follow-up, death indicator which are  features that should not be included in a realistic mortality predictor. The data is next inspected for highly correlated features and attributes that have a high correlation (>0.85) to another variable in the dataset are dropped. After this preprocessing is completed, the resulting dataframe to be used as input features for the models consists of 657736 rows and 23 columns. A numeric binary  variable indicating mortality/no mortality based on 11 years of follow-up, is defined separately as a target variable for the models. Initial exploratory data analysis is peformed to visualize the potential impact of different attributes on the mortality. 

***Modeling***

Two different machine learning approaches were used:

1) Unsupervised Learning: In this approach the population is divided into clusters using Principal Component Analysis and KMeans Cluster (n_cluster=3) based on the set of 23 input attributes. The clusters are evaluated to see if they are meaningful in terms of identifying groups that are at increased risk of all-cause mortality or mortality due to specific causes like cardiopulmonary diseases, cancer, infections, accidents.

2) Supervised Learning: In this approach, different machine learning models including Logistic Regression, Random Forest, Naive Bayes, Gradient Boosting Classifiers and Multi-Layer Perceptron (MLP) Classifier based on neural networks are evaluated and compared with their ROC-area under curve for prediction of mortality. The data is split into training and test sets using train_test_split with the default test_size of 0.25 and random_state set to 42. Since the data had a high degree of class imbalance with mortality in only 4.3% of the population in the database, the training data is oversampled using Synthetic Minority Oversampling Technique (SMOTE) to address the class imbalance. 

***Evaluation***

The unsupervised model is evaluated for meaningful differences in the groups created by the clustering method, based on demographic factors, overall mortality rates, incidences of mortality due to common causes like cardiopulmonary disease, cancer, infections, accidents. The supervised models are evaluated based on ROC-area under the curve (AUC) with a higher AUC in general indicating better performance in terms of sensitivity and specificity to mortality. Hold out cross-validation (using 'train test split' with a default test size of 0.25) is used to evaluate and compare the different classifier  models. Once the best model is identified, the relative feature importance is explored to determine the top attributes with the highest impact on model predictions. Counterfactuals are generated using the dice-ml package to determine changes in attributes that flip the model predictions of mortality in the test dataset.

#### Results

***Initial Exploratory Data Analysis***

It is clear from the initial exploratory analysis that the mortality group had a higher median age, lower education level , lower median income and a lower median number of household members (hhnum) compared to the alive (mortality=0) group. 

***Unsupervised Learning***

The unsupervised approach using PCA and KMeans Clustering with number of clusters =3 seem to have meaningfully divided the population into three clusters: cluster 2 with median age under 30 years and clusters 0 and 1 with median ages of under 40 and 45 years respectively. The two younger age groups, clusters 0 and 2, have higher percentages of people without health insurance compared to cluster 1. Interestingly, percentage of overall mortality is highest in cluster 0 even though it had the intermediate median age, closely followed by cluster 1 (which had the highest median age). Cluster 2 which is the youngest cluster, has the lowest mortality percentage as expected, though certain types of mortality like death due to infections had the highest incidence by percentage in cluster 2. Cluster 0 and cluster 1 have very similar (and higher, compared to cluster 2) percentages of mortality due to cardiopulmonary causes and cancer but incidence (%) of mortality due to accident in cluster 0 was close to two times that in clusters 1 and 2. While the clusters were interesting by some of their characteristics, the  mortality percentages in any one cluster were not high enough for any group to be considered highly sensitive and specific to overall mortality or mortality due to a specific cause.

***Supervised Learning***

The best model, based on the highest ROC-area under curve for supervised learning, turned out to be the Random Forest classifier with an ROC-AUC of 0.83 and an accuracy of 0.95 (even though accuracy alone as a metric is not as important here, given the high degree of class imbalance in the dataset). Based on the feature importance attribute of the Random Forest classifier, the top five factors with the most impact (in order of impact with the highest listed first) on the model predictions are the following:

1) age

2) hhnum : number of people in household

3) educ: education as measured by the level of highest grade completed

4) adjinc: inflation adjusted income

5) pob: region of birth

It is no surprise that age is the top factor for mortality prediction. However there are several socioeconomic factors like number of people living in household, level of education and income which were also quite impactful for the model prediction. Interestingly attributes like presence of social security (ssnyn), race, status of health insurance (histatus) had the least impact on model predictions. Counterfactuals generated using the dice-ml pacakage showed that an increase in the level of education and/or income would have resulted in opposite prediction for some cases where the original model prediction was positive for mortality.

**Conclusion**

Based on this model, investing in education early on, and enabling the population at high risk regions (based on region of birth) to complete a higher level of education and  have a higher income potential in future,  seems like a more effective strategy at preventing mortality and  extending life than enabling social security or health insurance. Number of people in household is also a top factor for mortality prediction, indicating the importance of providing in-person social/family support for preventing mortality and extending lives.

#### Next steps
The next step is deployment of the model. This can be potentially done with Amazon SageMaker which is a cloud machine-learning platform that enables developers to create, train, and deploy machine-learning models in the cloud. Based on the demographic, geographic and socio-economic factors that present a higher risk of mortality from this model, future studies can be designed to collect more data in the vulnerable population to better understand and develop a strategy that can help reduce mortality and extend  life span in this group.

#### Outline of project

- Link to notebook: https://github.com/SubhamGhosh12/UC_Berkeley_AI_ML/blob/main/Capstone/Capstone1a.ipynb



##### Contact and Further Information
Subham Ghosh, subham.ghosh@gmail.com
