% Naive bayes classifier
clc
clear all

%% --------------- Importing the dataset -------------------------
% ---------------------------- Code ---------------------------
data = readtable('train.csv');

classification_model_1 = fitcnb(data,'status', 'Distribution', 'kernel');

cv = cvpartition(classification_model_1.NumObservations, 'HoldOut', 0.4);
cross_validated_model_1 = crossval(classification_model_1,'cvpartition',cv); 

Predictions = predict(cross_validated_model_1.Trained{1},data(test(cv),1:end-1))
%%%testset to print test set data uncomment below line code
%data(test(cv),1:end-1)
confusionmatval = confusionmat(cross_validated_model_1.Y(test(cv)),Predictions);
TP=confusionmatval(1,1);
FN=confusionmatval(1,2);
FP=confusionmatval(2,1);
TN=confusionmatval(2,2);
Accuracy=((TP+TN)/(TP+TN+FP+FN))

