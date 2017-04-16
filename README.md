##Convolutional Neural Network (CNN) for HI-C Interaction Count Prediction##

###cnn.py###
Read in trainset data, build & save cnn model, evaluate against testset data, 
output plots.

Usage: python cnn.py \[chromosome #\] \[cell type\] \[dataset fold\] 
'''
python cnn.py 1 Gm12878 0
'''
###load.py###
Load existing cnn model, evaluate against testset data, output plots.

Usage: python cnn.py \[chromosome #\] \[cell type\] \[dataset fold\]
'''
python load.py 1 Gm12878 0
'''
###plots.py###	
Contains functions to output plots: Pearson correlation coefficient
by pairwise distance, histogram of interaction counts by pairwise
distance, mean/standard deviation for actual counts vs predicted
counts by pairwise distance, scatterplot of predicted vs actual
interaction counts.