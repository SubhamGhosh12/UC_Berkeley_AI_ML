# kraftwerk
This file contains a summary of findings from exploration of a dataset from the UCI Machine Learning repository. The data was collected via a survey on Amazon Mechanical Turk. The purpose of the exploration was to determine characteristics of drivers who are more likely to accept ***coffee house*** coupons.

# Brief description of the dataset
The dataset is based on a survey describing different driving scenarios, including the destination, current time, weather, passenger, etc., and then asking people whether they will accept the coupon if they are the driver. Answers given that the users will drive there “right away” or “later before the coupon expires” are labeled as “Y = 1”, and answers “no, I do not want the coupon” are labeled as “Y = 0”. There are five different types of coupons—less expensive restaurants (under $20), coffee houses, carry out and take away, bars, and more expensive restaurants ($20–$50). The current work focuses on identifying attributes that provide a higher likelihood of drivers accepting ***coffee house*** coupons.

# Codes for visualization and analysis
The codes used for visualization and analysis are included in the following notebook file with detailed step-by-step comments. The notebook can be found at this location:
https://github.com/SubhamGhosh12/kraftwerk/blob/main/prompt1.ipynb

A summary of the main findings from this analysis is included below.

# Summary
Total number of coffee-house coupons that were issued was 3816 and the overall acceptance rate was 0.496. The characteristics below seem to apply to drivers more likely to accept coffee house coupons:

1) drivers with age under 50 years: they have a better acceptance rate (0.509) compared to those above 50 ('50plus' group) where the acceptance rate is 0.42. The acceptance rate is most impressive for males in the age group below 21 (acceptance rate=0.819) compared to an acceptance rate of 0.483 for all others excluding males below 21

2) drivers with monthly coffee house visits of 1~3 times or more (acceptance rate=0.659 vs 0.34 for all others)

3) drivers with friend(s) or partners (acceptance rate = 0.591 vs 0.437 for all others)

4) drivers with no urgent destination (acceptance rate = 0.578 vs 0.401 for all others)

5) drivers driving in the same direction as the coffee house for which they receive coupons (acceptance rate = 0.527 vs 0.489 for not in the same direction)

6) drivers who are unemployed or students or people employed in healthcare support or as healthcare practitioners/technical (acceptance rate = 0.576 vs 0.465 for those in other occupations) 

7) drivers who are females and have 4 or more monthly visits to more expensive ($20-50) restaurants: they are more likely to accept coffee house coupons (acceptance rate=0.652) compared to all others (acceptance rate=0.491)

A few other factors which seem to influence acceptance rates are 

1) warm temperatures (80 degrees) associated with higher acceptance rates (0.527 vs 0.45 at colder temperatures)

2) time of the day: mid morning (10 am) and mid-afternoon (2 pm) times are associated with better acceptance rate  of 0.593 versus an acceptance rate of 0.425  earlier in the day (7 am) or later in the evening/night (6 pm, 10 pm)

3) longer expiry (1-day) coupons have higher acceptance rate of 0.581 compared to an acceptance rate of 0.429 for shorter expiry (2-hour) coupons



We  found that indeed when all five of the high-likelihood acceptance criteria  (age below 50, travelling with friend(s)/partner, number of monthly coffee house visits 1~3 times or more, 1 day expiration of coupons and time of the day being 10 AM or 2 PM) are simultaneously applied, the acceptance rate is very high (0.824) vs an acceptance rate of only 0.478 for all others. However, there are certain other age groups or demographics that have a high acceptance rate under certain conditions. For example,for drivers who reported their monthly frequency of coffee house visits as less than 1, coupons targeted to male drivers in this category may have a better acceptance rate than females. 

Three other important insights came out from the above analysis:

1. Even though acceptance rate of coupons on cold (30 degree) days is less (0.441), there are demographic groups with age over 50 years or male drivers below 21 years where acceptance rate of coupons is relatively high (0.604) on 30 degree days !


2. Acceptance rate of 2-hour coupons in general is low (0.429) as we found out earlier. However, for the demographic group of males below 21 years,acceptance of 2-hour coupons is relatively high (0.744)


3. Early morning (7 AM) acceptance rate of coffee house coupons is low (0.44). However, the 7 AM acceptance rate for the age group below 21 years is relatively high (0.694).


# Recommendations and Next Steps

### Which drivers should be targeted for coffee house coupons based on attributes associated with higher acceptance rates?
In general, coffee house coupons should be targeted to drivers below 50 years in age, drivers who are travelling with friend(s) or partner, who report 1 or more monthly visits to coffee houses. The coupons should have 1-day expiration and best time of the day for getting a higher acceptance rate is mid-morning (10 AM) or mid-afternoon (2PM). 

### Which attributes are associated with a low likelihood of acceptance of coffee-house coupons?
The chances of acceptance of coffee house coupons are the lowest in drivers who report 'never' in response to the query on number of monthly coffee house visits. Drivers who are alone have also a low chance of acceptance. Among the different age-groups, the 50 plus age group has the lowest acceptance rate. Chances of acceptance are lower for 2-hour expiration coupons compared to 1-day expiration, especially in females below 21 years of age who have the lowest acceptance rate of 2-hour expiration coupons.However, in comparison, the acceptance rate of 2-hour coupons for males below 21 years is markedly higher. Cold temperature (30F) is also associated, in general, with a lower acceptance of coupons for most drivers excluding males below 21 years and the 50 plus age group. Certain times of day, e.g. early in the morning (7 AM) or later in the evening/night (6 PM, 10 PM) are also associated with a lower likelihood of acceptance.

### Which drivers should be targeted for getting a better acceptance rate for certain contextual or coupon attributes that are generally associated with an overall lower acceptance rate? 
 On cold temperature (30 degree) days, coupons should be targeted to the following demographic groups: age over 50 or males below 21 years. Similarly, 2-hour expiry coupons should be targeted to the particular demographic group of males below 21 years. Coupons in the early morning (7 AM) should be targeted to the age group below 21 years  for a better acceptance rate at this time of the day. For drivers who report visiting coffee house less than once a month, coupons should be targeted to male drivers for a better acceptance rate.

