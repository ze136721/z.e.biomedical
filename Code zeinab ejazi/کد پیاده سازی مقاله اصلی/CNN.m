clc
clear
close all

%% Laod Data

load('./Data.mat');
label = categorical(label);

imageSize = [224 224];
N_data = size(Data,4);

for i = 1:N_data;
    img = Data(:,:,1,i);
    img2 = imresize(img,imageSize);
    Data1(:,:,1,i)=img2;
    Data1(:,:,2,i)=img2;
    Data1(:,:,3,i)=img2;
end

clear Data

%% Shuffling Data

ind_tr = round(0.75*N_data);

idx = randperm(N_data);

X = Data1(:,:,:,idx);
T = label(idx);

%% Train, Test and Validation Data Sepratation
X_Tr = X(:,:,:,1:ind_tr);
X_Ts = X(:,:,:,1+ind_tr:end);

T_Tr = T(1:ind_tr);
T_Ts = T(1+ind_tr:end);

clear seg_data1;

%% VGG16 
net = vgg16;

%% VGG16 feature extractor
layer = 'fc7';

for i = 1:N_data
    features(i,:) = activations(net,X(:,:,:,i),layer,'OutputAs','rows');
end

%% Train Network
classifier = fitcsvm(features,T,'Standardize',true,...
                    'KernelFunction','rbf','KernelScale','auto');

Y_Tr = predict(classifier,features(1:ind_tr,:));
Y_Ts = predict(classifier,features(1+ind_tr:end,:));
Y = predict(classifier,features);

T_tr = double(T_Tr)-1;
Y_tr = double(Y_Tr)-1;

T_ts = double(T_Ts)-1;
Y_ts = double(Y_Ts)-1;

T = double(T)-1;
Y = double(Y)-1;

%% Evaluation Network
CM_tr = confusionmat(T_tr,Y_tr');
CM_ts = confusionmat(T_ts,Y_ts');
CM = confusionmat(T,Y');

Train = evaluation(CM_tr)';
Test = evaluation(CM_ts)';
All = evaluation(CM)';

Table = table(Train,Test,All,'RowNames',{'Accuracy','Sensitivity','Specificity','Percision','F1-Score'});
save('.\Result\CNN.mat','Table')

plotconfusion(T_Tr,Y_Tr','Train',T_Ts,Y_Ts','Test',T,Y','All')
savefig('.\Result\CNN.fig')

