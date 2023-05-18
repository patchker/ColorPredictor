Predictive Models for gambling site for my own usage

This repository contains the implementation of various predictive models including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest and Gradient Boosting for predicting the next color in a popular online gambling site.

Prerequisites
The script requires Python 3.6+ and the following Python libraries installed:
NumPy
Selenium
Scikit-Learn
BeautifulSoup

Also, you need to have the webdriver for your preferred browser.

Usage
First, replace driver_path = "path/to/chromedriver" with the correct path to your webdriver in the main section of the code. This script opens a Selenium WebDriver to scrape gambling website, gets the history of previous colors, and then uses the predictive models to predict the next color.

The script features 4 different prediction models:

knn_predict(numbers): K-Nearest Neighbors. Prints the predicted color.
random_forest_predict(numbers): Random Forest. Prints the predicted color.
svc_predict(numbers): Support Vector Machine. Prints the predicted color.
gradient_boosting_predict(numbers): Gradient Boosting. Prints the predicted color.

Note
Gambling is addictive and can lead to severe losses. This script is for educational purposes only and should not be used for illegal activities.
