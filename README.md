# techChallenge

This is a Python script that is attempting to group documents in this dataset: http://csmining.org/index.php/webkb.html 

The program uses a TF*IDF approach to rate each word in the documents. It thus uses these rating as features in order to train a scikit provided Kmeans and Decision Trees models. The second one is clearly better than the Kmeans, but we can't accurately score the Kmeans due to the labels having different meaning.

# TODO

1. Implement Decision Trees
2. Implement Kmeans
3. Look into how to measure Kmeans score