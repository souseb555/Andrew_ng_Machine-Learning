# Machine Learning on Coursera
## Introduction
This repository contains all programming assignments of machine learning course taught by Andrew Ng.All the code base, data, screenshot, and images, are taken from, unless specified, [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning/)
## About the course
In this course, you will learn about the basics of  Machine Learning, understand how to build neural networks, and learn how to lead successful machine learning projects.The concepts are practised in MATLAB/OCTAVE.This is a 11 week Course and at the end of every week there will be a set of quizzes and Programming assignmemts(MATLAB/OCTAVE)
## OCTAVE
I have used OCTAVE-5.2.0_1-w64 version to implement programming assignments.You can download the same from [here](https://ftp.gnu.org/gnu/octave/windows/octave-5.2.0_1-w64-installer.exe).Dont worry if you are a complete beginner to OCTAVE /MATLAB, a complete tutorial on OCTAVE is found in Course itself.

## Folders

#### Linear Regression
In this Programming Assignment (ex1) you will implement linear regression and get to see it work on data.

- warmUpExercise.m
In the ﬁle warmUpExercise.m, you will ﬁnd the outline of an Octave/MATLAB function. Modify it to return a 5 x 5 identity matrix .

- computeCost.m
Basically computeCost.m is a function that computes J(θ).

- gradientDescent.m 
Here, you will implement gradient descent in the ﬁle gradientDescent.m. The loop structure has been written for you, and you only need to supply the updates to θ within each iteration. 
A good way to verify that gradient descent is working correctly is to look at the value of J(θ) and check that it is decreasing with each step.

- featureNormalize.m 
 Subtract the mean value of each feature from the dataset and  After subtracting the mean, additionally scale (divide) the feature values by their respective “standard deviations.”(std)
 
 - computeCostMulti.m and gradientDescentMulti.m 
 Here we implement the cost function and gradient descent for linear regression with multiple variables. 
 
- normalEqn.m 
Here we implemet the normal equation(θ =XTX−1 XT~y. ) and compute the value of theta .



#### Logistic Regression
In this Programming Assignment, you will implement logistic regression and apply it to two diﬀerent datasets. 

- sigmoid.m 
Here we implement a sigmoid function g(z) =g = 1./(1+exp(-z)).
##### Note: For a matrix, your function should perform the sigmoid function on every element

- costFunction.m 
Here we will compute cost and gradient for logistic regression

- predict.m
 The predict function will produce “1” or “0” predictions given a dataset and a learned parameter vector θ. 
 
 - costFunctionReg.m 
 Here you will implement code to compute the cost function and gradient for regularized logistic regression.
 
 - plotDecisionBoundary.m 
 To help you visualize the model learned by this classiﬁer, we have provided the function plotDecisionBoundary.m which plots the (non-linear) decision boundary that separates the positive and negative examples
 
 -  mapFeature.m
 One way to ﬁt the data better is to create more features from each data point. In the provided function mapFeature.m, we will map the features into all polynomial terms of x1 and x2 up to the sixth power



#### Multi-class Classification and Neural Networks
In this Programming Assignment, you will implement one-vs-all logistic regression and neural networks to recognize hand-written digits.

- lrCostFunction.m 
computes the cost using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters.

- oneVsAll.m
ONEVSALL trains multiple logistic regression classifiers and returns all the classifiers in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier for label i.

- predictOneVsAll.m
Predict the label for a trained one-vs-all classifier. The labels are in the range 1..K, where K = size(all_theta, 1).

- predict.m
p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2).


#### Neural Network Learning
In this Programming Assignment, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.

- sigmoidGradient.m :
SIGMOIDGRADIENT returns the gradient of the sigmoid function

- randInitializeWeights.m:
Randomly initialize the weights of a layer with L_in for incoming connections and L_out for outgoing connections.

- nnCostFunction.m:
Implements the neural network cost function for a two layer neural network which performs classification.



#### Regularized Linear Regression and Bias_Variance
In this Programming Assignment, you will implement regularized linear regression and use it to study models with diﬀerent bias-variance properties. 

- linearRegCostFunction.m:
Compute cost and gradient for regularized linear regression with multiple variables.

- learningCurve.m- polyFeatures.m :
Generates the trainning set errors and cross validation set errors required to plot a learning curve.

- polyFeatures.m :
Maps X (1D vector) into the p-th power.

- validationCurve.m :
Generate the train and validation errors needed to plot a validation curve that we can use to select lambda.



#### Support Vector Machines
In this Programming Assignment, you will be using support vector machines (SVMs) to build a spam classiﬁer.

- gaussianKernel.m :
This function returns a radial basis function kernel between x1 and x2.

- dataset3Params.m :
DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise where you select the optimal (C, sigma) learning parameters to use for SVM with RBF kernel.

- processEmail.m :
PROCESSEMAIL preprocesses a the body of an email and returns a list of word_indices.

- emailFeatures.m :
EMAILFEATURES takes in a word_indices vector and produces a feature vector from the word indices.

#### K-Means Clustering and PCA
In this Programming Assignment, you will implement the K-means clustering algorithm and apply it to compress an image.

- pca.m :
PCA Run principal component analysis on the dataset X.

- projectData.m :
PROJECTDATA Computes the reduced data representation when projecting only on to the top k eigenvectors.

- recoverData.m :
RECOVERDATA Recovers an approximation of the original data when using the projected data.

-findClosestCentroids.m :
This function takes the data matrix X and the locations of all centroids inside centroids and should output a one-dimensional array idx that holds the index (a value in {1,...,K}, where K is total number of centroids) of the closest centroid to every training example.

- computeCentroids.m :
COMPUTECENTROIDS returns the new centroids by computing the means of the data points assigned to each centroid.

- kMeansInitCentroids.m :
KMEANSINITCENTROIDS This function initializes K centroids that are to be used in K-Means on the dataset X
randperm :Randomly reorder the indices of examples. Then, it selects the ﬁrst K examples based on the random permutation of the indices. This allows the examples to be selected at random without the risk of selecting the same example twice.


#### Anomaly Detection and Recommender Systems

In this Programming Assignment, you will implement the anomaly detection algorithm and apply it to detect failing servers on a network.

- estimateGaussian.m :
ESTIMATEGAUSSIAN This function estimates the parameters of a Gaussian distribution using the data in X.

- selectThreshold.m
SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting:outliers.

- cofiCostFunc.m :
In this part of the exercise, you will implement the collaborative ﬁltering learning algorithm and apply it to a dataset of movie ratings.

#### Also Note!
> You can make use of the Discussion forums in the course so as to resolve any errors you face while doing assignments.
>You may also take help from the PDF files attached in each folder.
>After completing a part of the exercise, you can submit your solutions for grading by typing submit at the Octave/MATLAB command line. The submission script will prompt you for your login e-mail and submission token and ask you which ﬁles you want to submit. You can obtain a submission token from the web page for the assignment.


##You may also refer to
- https://www.geeksforgeeks.org/ml 
- https://www.youtube.com/c/AladdinPersson/videos
- https://www.apdaga.com/2020/01/coursera-machine-learning-all-weeks-solutions





