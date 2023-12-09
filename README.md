#  ML algorithm implementation
## This is a machine learning library developed by Yufeng Yang for CS5350/6350 in University of Utah
To run the code, run **run.sh** in the corresponding folder with the following command:
chmod 755 run.sh \
./run.sh
**Note**
You actually only need to run **run.sh** once in each sub folder in Ensemble Learning. \
The results is stored in the test file name in the format like: bank_test_xxxx.py 
### Neural Network
NeuralNetwork.py implements all the APIs of neural nets, such as forward propogation, backword propogation,SGD etc.\
bank_test.py is test case for bank note dataset.\
dnn.py is pytorch version of neural nets. Please ensure to install pytorch, torchvision before running the code.
### logistic regression
logistic.py includes SGD for MAP and MLE version logistic regression.\
bank_test.py is test case for dataset.
### SVM
All apis of SVM: linear primal dual, kernel svm and bonus question (kernel perceptron) are in SupportVectorMachine.py
bank_test.py loads data and hyper-parameter for test.
### Perceptron
The test case are all stored in bank_test.py, to run the result, run the run.sh file. All APIs of standard, voted, average perceptron \
are in perceptron.py
### Adaboost
In bank_test_ada.py, T represents the number of subtrees,depth is the maximal depth for single decision tree. 
### Bagging
In bank_test_bagging.py, T represents the number of subtrees,depth is the maximal depth for single decision tree. \
sample_size is the training sample size we want.
In bank_test_bvb.py, same for T and depth, num_run is the total iteration we want to run, sample_size now is 1000. \
sample_train_data_round_size is the size of the training data size of bagged tree. 
### Random Forest
In bank_test_rf.py, same meaning as above for T and depth, subset_size is the sub-feature set size we want to sample. \
In benk_test_rfbv.py, same meaning for T, num_run, depth,sample_size, the sample size for feature set is in the line: \
RF.ID3(metric_selection = 'entropy', max_depth=depth, attribute_subset = 2).
### Linear Regression
The test case is stored in concrete_test.py.\
change the iteration by setting the variable LMS.max_iter \
change the learning rate by LMS.lr
