Jinu Jacob
Linear Regression

----------------------------------------------------------------
                            CFLR.m
----------------------------------------------------------------
Performs closed form linear regression on a data set


HOW TO RUN:

To run from the Matlab command line use:

	CFLR()

The default file used will be 'x06Simple.csv'.
To use a different file use:

	CFLR(file name)



----------------------------------------------------------------
			    sFoldsCrossVal.m
----------------------------------------------------------------
Performs s-folds cross-validation on a data set


HOW TO RUN:

To run from the Matlab command line use:
	
	sFoldsCrossVal()

The default file used will be 'x06Simple.csv' and the default s value will be 5.
To use a different file and s value use:

	sFoldsCrossVal(file name, s)



----------------------------------------------------------------
			gradientDescent.m
----------------------------------------------------------------
Performs batch gradient descent on a data set


HOW TO RUN:

To run from the Matlab command line use:
	
	gradientDescent()

The default file used will be 'x06Simple.csv'.
To use a different file use:

	gradientDescent(file name)



Note: The full file path should be specified unless the path has been added to Matlabs's pathdef.m
