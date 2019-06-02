# IndiaMart-Hackathon-2k19
# Objective
Provide a working prototype solution to gauge the appropriate unit wise price range for the 3 categories based on their units by removing outliers from the data and we can split it into 2 parts.
* Extracting Data from the given file and finding and removing outliers from it
* Find a PDF that fits the given data and suggests an appropriate range from it.

We are trying to find PDF because the price of the product can be regarded as a continuous variable.
# Idea
For removing outliers first we are filtering entries whose units are not recognized for which we can define a list of valid units and then compare every unit of the data with that list. For removing the outliers whose price deviates too much we are using Z-score which tells how much an entry deviates from the mean and remove all entries whose Z-score is greater (or less) than 3(-3) which is a generally used threshold.

For finding Probability density function we are using **Kernel Density Estimation** which is useful to find PDF from discrete data. We are using Gaussian Kernel with normal distribution approximation then we are using Bayes theorem to eliminate part of PDF  whose price < 0 as the probability of an item having the price less than 0  is 0. Finally, we are using the above-calculated PDF to find the smallest range which has probability of an item priced in it is greater than 0.5 or 0.75(adjustable parameter).
# Implementation
We have converted a single.xlsx file to 3 .csv file(one for each item) then used **pandas** library to extract and remove outliers, we used **scipy** library to find PDF, **seaborn** library for visualization. We implemented our code in Jupyter notebook which is included in the folder we have also added an explanation to the implementation in it. We also added a .py file which gives all the PDF graphs and ranges obtained by us.

**ZIP file contains:**

1) an .ipynb file (to visualize or to change parameters)

2) a .pdf file (which is our final report

3) a .py file (for direct running)

4) three .csv files (which contains data)
