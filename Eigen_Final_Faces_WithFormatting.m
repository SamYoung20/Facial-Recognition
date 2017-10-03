tic
%% LOAD IN THE TEST DATA
load face_detect.mat;
test_images = faces_train;
size(test_images);
T = test_images(:,:, 1);
clf
dataset2 = faces_test_hard;
dataset1 = faces_test_easy;
names2 = names_test_hard;
%%

%% Reshape into the column vectors for each images
facesStacked = reshape(test_images,size(test_images,1)*size(test_images,2),size(test_images,3));
size(facesStacked);
%%

%% Find Mean Face
sumFace = [];
for i = 1: 65536
    sumFace(i,1) = sum(facesStacked(i,:));
end    
meanFace =sumFace./size(test_images,3);
%%

%% Test Mean Face
meanTest = reshape(meanFace,256, 256);
imagesc(meanTest); colormap('gray');
%%

%%Recenter Faces vs mean Face
centeredFaces = facesStacked;
for i = 1: size(facesStacked,2)
    centeredFaces(:,i) = facesStacked(:,i) - meanFace;
end  
meanFace;
centeredFaces;
%%

%%  Make Covariace Matric and Calculating Eigenface
A = centeredFaces;
M = A'*A;
[Vi,eigValues]= eig(M);

%Ui are the eigen faces
for i = 1: size(test_images,3)
    Ui(:,i) = A*Vi(:,i);
    Ui(:,i) = Ui(:,i)./ norm(Ui(:,i));
end
Ui;
testU = reshape(Ui(:,246),256, 256); % the largest eigenvalue's corresponding eigenface normalized
imagesc(testU); colormap('gray');
%%
%%Select top most signigicant faces
topX = 30;
K = Ui(:,246-topX +1:246);
%%
toc
tic
%% Find a Random Face and guess which face it is
idx = randi(size(dataset2,3));
nameofRandom = names2(:,idx)'
randomOriginal = dataset2(:,:,idx);
%reshape image to column vector
randomFace = reshape(dataset2(:,:,idx),size(dataset1(:,:,idx),1)*size(dataset1(:,:,idx),2),1);
%center data on mean
centeredRandomFace = randomFace - meanFace;
weightsRando = [];
for i =1:topX
    weightsRando(i,1) = K(:,i)'*centeredRandomFace;
end
weightsTraining = [];
for i =1:topX
    weightsTraining(i,:) = K(:,i)'*centeredFaces;
end
weightsRando;
weightsTraining;
dif = 100000;
minimum = dif;
minIndex = 10000;
for i =1 : size(weightsTraining,2)
    dif = norm(weightsRando - weightsTraining(:,i));
    if dif < minimum
        minimum = dif;
        minIndex = i;
    end
end
minimum;
minIndex;
Guess = reshape(facesStacked(:,minIndex),256, 256); % the largest eigenvalue's corresponding eigenface normalized
guessName = names_train(:,minIndex)'
%%
toc
%%PLOT ALL THE FACES
subplot(2,1,1), imagesc(randomOriginal); colormap('gray');
axis equal
xlim([0 256]);
ylim([0 256]);
title(strcat('Actual Image:', {'   '}, nameofRandom) )
set(gca,'YTick',[])
set(gca,'XTick',[])
subplot(2,1,2), imagesc(Guess); colormap('gray');
axis equal
xlim([0 256]);
ylim([0 256]);
title(strcat('Genie Guess:',{'  '}, guessName) )
set(gca,'YTick',[])
set(gca,'XTick',[])
%%