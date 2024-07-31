

%% training model & validation --> naive bayse model
% K-fold
K = 10;
n_run = 10;
accuracy = zeros(K,n_run);
cl = cl+1;
% 10_fold
for i_run=1:n_run
    indices = crossvalind('Kfold',y_train,K);
    
    for i_fold = 1:K
        Val = indices==i_fold;
        Train = ~Val;
        featureTrain = x_train(:,Train);
        featureVal = x_train(:,Val);
        
        % Classification
        % KNN
        KNNModel = fitcknn(featureTrain',y_train(Train));
        
        % validation
        class = predict(KNNModel, featureVal');
        accuracy(i_fold,i_run) = 100*length(find(class' == y_train(Val)))/length(y_train(Val));
        
        valCKNN = crossval(KNNModel);
        classerrorKNN(i_fold,i_run) = kfoldLoss(valCKNN)*100;

    end
    disp(['n_run = ',num2str(i_run),', Accuracy = ',num2str(mean(accuracy(:,i_run))),' ± ',num2str(std(accuracy(:,i_run)))])
    disp(['n_run = ',num2str(i_run),', ClassError = ',num2str(mean(classerrorKNN(:,i_run))),' ± ',num2str(std(classerrorKNN(:,i_run)))])
end
disp(['Total Accuracy KNN = ',num2str(mean(accuracy(:))),' ',native2unicode(177),' ',num2str(std(accuracy(:)))]);
disp(['Total ClassError KNN = ',num2str(mean(classerrorKNN(:))),' ',native2unicode(177),' ',num2str(std(classerrorKNN(:)))]);

%% Test KNN
y_tr_predictKNN = predict(KNNModel, x_train');
y_ts_predictKNN = predict(KNNModel, x_test');

%% Confusion matrix KNN

[c_tr order] = confusionmat(y_train,y_tr_predictKNN');
TN=c_tr(1,1);
FP=c_tr(2,1);
FN=c_tr(1,2);
TP=c_tr(2,2);

acc(cl,1) = ((TP+TN)/sum(sum(c_tr)))*100;
sen(cl,1) = TP/(TP+FN) *100;  % TPR or recal
spe(cl,1) = (TN/(TN+FN)) *100 ;

[c_ts order] = confusionmat(y_test,y_ts_predictKNN');
TN=c_ts(1,1);
FP=c_ts(2,1);
FN=c_ts(1,2);
TP=c_ts(2,2);

acc(cl,2) = ((TP+TN)/sum(sum(c_ts)))*100;
sen(cl,2) = TP/(TP+FN) *100;  % TPR or recal
spe(cl,2) = (TN/(TN+FN)) *100 ;



