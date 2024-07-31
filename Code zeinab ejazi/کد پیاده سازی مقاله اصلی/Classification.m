clc
clear
close all

load('feature.mat');

cl = 0;

[tr,~,ts] = dividerand(numel(label),0.7, 0, 0.3);

x_train = feature(:,tr);
y_train = label(tr);

x_test = feature(:,ts);
y_test = label(ts);

disp('*********************************************')

disp('=============================================')
disp('KNN : ')
disp(' ')
classifier_KNN
disp('=============================================')
disp('SVM : ')
disp(' ')
classifier_SVM
disp('=============================================')
disp('Decision Tree : ')
disp(' ')
classifier_decision_Tree
disp('=============================================')

CM_tr = confusionmat(y_train,y_tr_predictKNN');
CM_ts = confusionmat(y_test,y_ts_predictKNN');
Train_KNN = evaluation(CM_tr)';
Test_KNN = evaluation(CM_ts)';
          
CM_tr = confusionmat(y_train,y_tr_predictSVM');
CM_ts = confusionmat(y_test,y_ts_predictSVM');
Train_SVM = evaluation(CM_tr)';
Test_SVM = evaluation(CM_ts)';

CM_tr = confusionmat(y_train,y_tr_predictDT');
CM_ts = confusionmat(y_test,y_ts_predictDT');
Train_DT = evaluation(CM_tr)';
Test_DT = evaluation(CM_ts)';

row = {'Accuracy','Sensitivity','Specificity','Percision','F1-Score'};

Table = table(Train_KNN,Train_SVM,Train_DT,...
              Test_KNN,Test_SVM,Test_DT,'RowNames',row);

figure
plotconfusion(y_train,y_tr_predictKNN','KNN Train',...
              y_train,y_tr_predictSVM','SVM Train',...
              y_train,y_tr_predictDT','SVM Train',...
              y_test,y_ts_predictKNN','KNN Test',...
              y_test,y_ts_predictSVM','SVM Test',...
              y_test,y_ts_predictDT','SVM Test')

disp('Result')
disp(Table);
disp('========================================================')
















