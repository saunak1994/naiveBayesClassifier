
NAIVE BAYES TEXT CLASSIFIER PACKAGE
============================================

This package implements a Naive-Bayesian classifier that classifies text documents into 20 categories based on 
Machine learning. The original data set is available at http://qwone.com/~jason/20Newsgroups/. In this package,
a processed version of the data set has been used. The package has been written using Python 3.6 and uses the NumPy
Library to structure the input data .

HOW TO RUN THE PACKAGE
============================================

1. Make sure you have Python installed in your system and the NumPy library has been included and updated. 

2. Have the 20newsgroups files in the same folder as the script naiveBayesClassifier.py 

3. run the code on the terminal (or cmd) as follows: 


$python naiveBayesClassifier.py vocabulary.txt map.csv train_label.csv train_data.csv test_label.csv test_data.csv testDataType estimatorType > outputFile.txt


	1. testDataType should be 0 if you want to test on training data. In this case, estimatorType is don't care. It will always test on 
	Bayesian Estimators

	2. testDataType should be 1 if you want to test on testing data. In this case, estimatorType should be 0 for Bayesian Estimators and 1
	for Maximum-Likelihood Estimators. 

for example: if you want to evaluate on testing data using Bayesian Estimators, your command should be:


$python naiveBayesClassifier.py vocabulary.txt map.csv train_label.csv train_data.csv test_label.csv test_data.csv 1 0 > outputFile.txt




©Saunak Saha 
====================
(saha@iastate.edu)