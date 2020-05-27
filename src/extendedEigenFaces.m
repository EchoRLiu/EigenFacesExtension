close all; clear all; clc

% Load cropped faces dataset.
m1=192; n1=168; % Resolution of images.
nw1=96*84; % SVD resolutions.
cropped_faces=zeros(32256, 2432); 
% we got 2432 images in croppedYale folder, we are missing yaleB14 folder.

file='/Users/yuhongliu/Downloads/CroppedYale/yaleB';
pos=0;

for i=1:39
    if i<=9
        dirt=append(file,string(0),string(i));
    else
        dirt=append(file,string(i));
    end
    file_ls=dir(dirt);
    NF = length(file_ls);
    for j = 3:NF
        pos=pos+1;
        cropped_faces(:, pos) = reshape(imread(fullfile(dirt, file_ls(j).name)), 32256,1);
    end
end
% We have 64 images from 38 people.

%%

% Load uncropped faces dataset.
m2=243; n2=320; nw2=122*160;% remember the shape of images.
uncropped_faces=zeros(77760,165); % we have 15 subjects with 11 different faces.

file='/Users/yuhongliu/Downloads/yalefaces_uncropped/yalefaces/subject';
subtitle=[".centerlight",".glasses",".happy",".leftlight",".noglasses",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"];

for i=1:15
    for j=1:11
        if i<=9
            file_title=append(file,string(0),string(i),subtitle(j));
        else
            file_title=append(file,string(i),subtitle(j));
        end
        pos=11*(i-1)+j;
        uncropped_faces(:,pos)=reshape(imread(file_title), 77760, 1);
    end
end

%%

% Get the wavelet representations.
cropped_faces_wave = dc_wavelet(cropped_faces, m1, n1, nw1);
uncropped_faces_wave = dc_wavelet(uncropped_faces, m2, n2, nw2);

feature = 60;

%%

[U1,S1,V1]=svd(cropped_faces_wave,0);
U1=U1(:,1:feature);
[U2,S2,V2]=svd(uncropped_faces_wave,0);
U2=U2(:,1:feature);

%%

% The first four POD modes.
figure(1);
for j = 1:4
    % Cropped faces.
    subplot(4,2,2*j-1);
    ut1=reshape(U1(:,j), 96, 84);
    ut2=ut1(96:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Cropped Eigenfaces with mode' num2str(j)]);
    % Uncropped faces.
    subplot(4,2,2*j);
    ut1=reshape(U2(:,j), 122, 160);
    ut2=ut1(122:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Uncropped Eigenfaces with mode' num2str(j)]);
end

%%

figure(2);
% Cropped faces.
subplot(2,2,1); % normal scale.
plot(diag(S1)/sum(diag(S1)),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('Energy')
title('Normal scale: Cropped')
subplot(2,2,3);
semilogy(diag(S1)/sum(diag(S1)),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('Energy')
title('Log scale: Cropped')
% Uncropped faces.
subplot(2,2,2); % normal scale.
plot(diag(S2)/sum(diag(S2)),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('Energy')
title('Normal scale: Uncropped')
subplot(2,2,4);
semilogy(diag(S2)/sum(diag(S2)),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('Energy')
title('Log scale: Uncropped')

%%

energy1 = diag(S1)/sum(diag(S1));
thresh90 = 0;
for i = 1:2432
    thresh90 = thresh90 + energy1(i);
    if thresh90 > .9
        disp(i) % rank can be 1666.
        break
    end
end

energy2 = diag(S2)/sum(diag(S2));
thresh90 = 0;
for i = 1:165
    thresh90 = thresh90 + energy2(i);
    if thresh90 > .9
        disp(i) % rank can be 126.
        break
    end
end

%%

% Projection of the first 40 face images onto the first four POD modes.
figure(3);
for j=1:4
    % Cropped faces.
    subplot(4,2,2*j-1);
    plot(1:40,V1(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 Cropped images Projection onto mode' num2str(j)])
    % Uncropped faces.
    subplot(4,2,2*j);
    plot(1:40,V2(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 Uncropped images Projection onto mode' num2str(j)])
end

% should we use u or v as our projection.

%%

% Inviduals.
figure(4)
for i = 1:38
    plot3(V1((i-1)*64+1:i*64,1), V1((i-1)*64+1:i*64, 2), V1((i-1)*64+1:i*64, 3), 'o'),
    hold on
    title('Visualisation of individual clusters for Cropped Images');
end

figure(5)
for i = 1:15
    plot3(V2((i-1)*11+1:i*11,1), V2((i-1)*11+1:i*11, 2), V2((i-1)*11+1:i*11, 3),'o'),
    hold on
    title('Visualisation of individual clusters for Uncropped Images');
end

%%

fm = [5 26 27 31 33 36];
% Female vs Male.
figure(6)
for i = 1:38
    if ismember(i, fm)
        plot3(V1((i-1)*64+1:i*64,1), V1((i-1)*64+1:i*64, 2), V1((i-1)*64+1:i*64, 3), 'ro'),
        hold on
    else
        plot3(V1((i-1)*64+1:i*64,1), V1((i-1)*64+1:i*64, 2), V1((i-1)*64+1:i*64, 3), 'bo'),
        hold on
    end
    title('Visualisation of female vs male clusters for Cropped Images');
end

%% Test 1. Classify individuals.

avg_correct=0;
cross=300; % Cross-validation is important!!
ctest = [];
ctrain = [];
for j = 1:38
    ctrain=[ctrain; j*ones(57,1)];
    ctest=[ctest; j*ones(7,1)];
end

% un = 1; feature = 20; % GM method.
un = 0; feature=40; % Other supervised methods.
if un == 1
    
    i = 1;
    while i <= cross

        % figure(4+i)

        % Seperate training and testing dataset.
        % use around 90% of images for training.

        qs = zeros(64, 38);
        for j = 1:38
            qs(:,j) = randperm(64);
        end

        xtrain = zeros(2166,feature);
        xtest = zeros(266,feature);
        for k = 1:38
            individual = V1((64*(k-1)+1):(64*k),1:feature);
            q = qs(:,k);
            xtrain((57*(k-1)+1):(57*k),:) = individual(q(1:57),:);
            xtest((7*(k-1)+1):(7*k),:) = individual(q(58:end),:);
        end

        % GM has very low accuracy. 2.58%.
        try
            gm=fitgmdist(xtrain,38);
            pre=cluster(gm, xtest);
            correct=sum((pre == ctest));
            avg_correct = avg_correct + correct/266;
            disp(i)
            i = i+1;
        catch
            i = i;
        end
    end
  
else
    
    for i = 1:cross

        % figure(4+i)

        % Seperate training and testing dataset.
        % use around 90% of images for training.

        qs = zeros(64, 38);
        for j = 1:38
            qs(:,j) = randperm(64);
        end
        xtrain = zeros(2166,feature);
        xtest = zeros(266,feature);
        for k = 1:38
            individual = V1((64*(k-1)+1):(64*k),1:feature);
            q = qs(:,k);
            xtrain((57*(k-1)+1):(57*k),:) = individual(q(1:57),:);
            xtest((7*(k-1)+1):(7*k),:) = individual(q(58:end),:);
        end

    %     % Naive Bayesian method. 82.11%
        nb=fitcnb(xtrain, ctrain);
        pre=nb.predict(xtest);

        % Linear Discrimination behave the best with spectrogram.
        % 86.48%.
        % pre=classify(xtest, xtrain, ctrain);

        % SVM. 68.16%
%         svm=fitcecoc(xtrain,ctrain);
%         pre=predict(svm, xtest);

        % bar(pre);
        correct=sum((pre == ctest));
        avg_correct = avg_correct + correct/266;
    end
end

disp(avg_correct/cross);

%% Test 2. Gender Classification.

avg_correct=0;
cross=300; % Cross-validation is important!!

fm = [5 26 27 31 33 36]; % [5 27-1 28-1 32-1 34-1 37-1]; 
% Missing folder 14. Hence the -1.
% Label Male as 1, female as 2.
ctest = ones(266,1);
ctrain = ones(2166,1);
for j = 1:6
    ctrain((57*(fm(j)-1)+1):(57*fm(j)), 1) = 2;
    ctest((7*(fm(j)-1)+1):(7*fm(j)), 1)= 2;
end
feature = 100;

un = 1; % GM method.
%un = 0; % Other supervised methods.

if un == 1
    
    i = 1;
    while i <= cross

        % figure(4+i)

        % Seperate training and testing dataset.
        % use around 90% of images for training.
        qs = zeros(64, 38);
        for j = 1:38
            qs(:,j) = randperm(64);
        end
        xtrain = zeros(2166,feature);
        xtest = zeros(266,feature);
        for k = 1:38
            individual = V1((64*(k-1)+1):(64*k),1:feature);
            q = qs(:,k);
            xtrain((57*(k-1)+1):(57*k),:) = individual(q(1:57),:);
            xtest((7*(k-1)+1):(7*k),:) = individual(q(58:end),:);
        end

        % GM has low accuracy. 51.62%.
        try
            gm=fitgmdist(xtrain,2); % Female vs Male.
            pre=cluster(gm, xtest);
            correct=sum((pre == ctest));
            avg_correct = avg_correct + correct/266;
            disp(i)
            i = i+1;
        catch
            i = i;
        end
    end
  
else
    
    for i = 1:cross

        % figure(4+i)

        % Seperate training and testing dataset.
        % use around 90% of images for training.

        qs = zeros(64, 38);
        for j = 1:38
            qs(:,j) = randperm(64);
        end

        xtrain = zeros(2166,feature);
        xtest = zeros(266,feature);
        for k = 1:38
            individual = V1((64*(k-1)+1):(64*k),1:feature);
            q = qs(:,k);
            xtrain((57*(k-1)+1):(57*k),:) = individual(q(1:57),:);
            xtest((7*(k-1)+1):(7*k),:) = individual(q(58:end),:);
        end
        
        % KNN.
        [ind, D] = knnsearch(xtrain, xtest, 'k', 2);

    %     % Naive Bayesian method. 85.39%
%         nb=fitcnb(xtrain, ctrain);
%         pre=nb.predict(xtest);

        % Linear Discrimination behave the best with spectrogram.
        % 92.27%.
        pre=classify(xtest, xtrain, ctrain);

        % SVM. % 84.36%
        svm=fitcecoc(xtrain,ctrain);
        pre=predict(svm, xtest);

        % bar(pre);
        correct=sum((pre == ctest));
        avg_correct = avg_correct + correct/266;
    end
end

disp(avg_correct/cross);

%%

function dcData = dc_wavelet(dcfile, m, n, nw)
    [p,q]=size(dcfile);
    nbcol = size(colormap(gray),1);
    dcData=zeros(nw,q);
    
    for i = 1:q
        X=double(reshape(dcfile(:,i),m,n));
        [cA,cH,cV,cD]=dwt2(X,'haar');
        cod_cH1 = wcodemat(cH,nbcol);
        cod_cV1 = wcodemat(cV,nbcol); 
        cod_edge=cod_cH1+cod_cV1; 
        dcData(:,i)=reshape(cod_edge,nw,1);
    end
end
