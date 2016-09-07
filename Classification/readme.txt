Jinu Jacob
Classification
Datasets: 
KNN, Naive Bayes, PR Tradeoff, Logistic Regression, Builtins: https://archive.ics.uci.edu/ml/datasets/Spambase 

Multi-Class Classification:
http://archive.ics.uci.edu/ml/datasets/Cardiotocography
2


----------------------------------------------------------------
                            KNN.m
----------------------------------------------------------------
Performs k-Nearest Neighbors classification on data set


HOW TO RUN:

To run from the Matlab command line use:

	KNN()

The default file used will be 'spambase.data'.and the default k value will be 5.
To use a different file and s value use:

	KNN(filename, k)

Dataset source: https://archive.ics.uci.edu/ml/datasets/Spambase 


----------------------------------------------------------------
			    naiveBayes.m
----------------------------------------------------------------
Builds a Naive Bayes Classifier


HOW TO RUN:

To run from the Matlab command line use:
	
	naiveBayes()

The default file used will be 'spambase.data' to use a different file use:
	
	naiveBays(filename) 



----------------------------------------------------------------
			prTradeoff.m
----------------------------------------------------------------
Plots precision-recall tradeoff for varying thresholds in Naive Bayes


HOW TO RUN:

To run from the Matlab command line use:
	
	prTradeoff()

The default file used will be 'spambase.data'.
To use a different file use:

	prTradeoff(file name)



Note: The full file path should be specified unless the path has been added to Matlabs's pathdef.m
