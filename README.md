# Technical Challenge

This is a Python script that is attempting to group documents in this dataset: http://csmining.org/index.php/webkb.html 

The program uses a TF*IDF approach to rate each word in the documents. It thus uses these rating as features in order to train a scikit provided Kmeans, Logistic Regression and Decision Trees models.

# Results
The results are as follow:
1. Logistic Regression:
	Accuracy on the training subset: 0.995
	Accuracy on the test subset: 0.872
2. Decision Trees:
	Accuracy on the training subset: 0.995
	Accuracy on the test subset: 0.753
3. KMeans (4):
	Accuracy on the training subset: 0.352


The script is running all this algorithms sequencially. To run it simply type: python solution.py

PS: the solution uses python 2.7 in order to run. I think the only change required for 3.5 is adding () to the print function.
