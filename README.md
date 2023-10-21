# Decision Tree Implementation Yufeng
## This is a machine learning library developed by Yufeng Yang for CS5350/6350 in University of Utah
To run the code, run **run.sh** in the corresponding folder with the following command:
chmod 755 run.sh \
./run.sh
**Note**
You actually only need to run **run.sh** once in each sub folder in Ensemble Learning. \
The results is stored in the test file name in the format like: bank_test_xxxx.py 
### Adaboost
In bank_test_ada.py, T represents the number of subtrees,depth is the maximal depth for single decision tree. \
### Bagging
In bank_test_bagging.py, T represents the number of subtrees,depth is the maximal depth for single decision tree. \
sample_size is the training sample size we want.
In bank_test_bvb.py, same for T and depth, num_run is the total iteration we want to run, sample_size now is 1000. \
sample_train_data_round_size is the size of the training data size of bagged tree. 
### Random Forest
In bank_test_rf.py, same meaning as above for T and depth, subset_size is the sub-feature set size we want to sample. \
In benk_test_rfbv.py, same meaning for T, num_run, depth,sample_size, the sample size for feature set is in the line: \
RF.ID3(metric_selection = 'entropy', max_depth=depth, attribute_subset = 2).
