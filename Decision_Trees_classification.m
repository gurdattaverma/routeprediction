% Decision Tree Classifier
clc
clear all
%% --------------- Importing the dataset -------------------------
% ---------------------------- Code ---------------------------
data = readtable('train.csv');

classification_model = fitctree(data,'status');

%% -------------- Test and Train sets ----------------------------
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);
cross_validated_model = crossval(classification_model,'cvpartition',cv); 


%% -------------- Making Predictions for Test sets ---------------
% ---------------------------- Code ---------------------------
%%%testset to print test set data uncomment below line code
%data(test(cv),1:end-1)
Predictions = predict(cross_validated_model.Trained{1},data(test(cv),1:end-1))


%% -------------- Analyzing the predictions ---------------------
% ---------------------------- Code ---------------------------

confusionmatval = confusionmat(cross_validated_model.Y(test(cv)),Predictions);
TP=confusionmatval(1,1);
FN=confusionmatval(1,2);
FP=confusionmatval(2,1);
TN=confusionmatval(2,2);
Accuracy=((TP+TN)/(TP+TN+FP+FN))
view(classification_model,'mode','graph') 
