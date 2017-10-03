tic
%% 1. Load in Test Data
load face_detect.mat
test_images = faces_train;
size(test_images);
T = test_images(150:225,50:200, 1);
test_images_mouth= test_images(150:225,50:200,:);
size(test_images_mouth)
axis equal

%% 2. Reshape into column vectors for each image.
facesStacked = reshape(test_images_mouth ,size(test_images_mouth,1)*size(test_images_mouth,2),size(test_images_mouth,3));

%% 3. Find Mean Face
sumFace = [];
for i = 1: size(facesStacked,1)
    sumFace(i,1) = sum(facesStacked(i,:));
end    
meanFace = sumFace./size(test_images_mouth,3);

%% 4. Test Mean Face
meanTest = reshape(meanFace,size(test_images_mouth,1),size(test_images_mouth,2));
axis equal
% imagesc(meanTest); colormap('gray');

%% 5. Recenter Faces vs mean Face
centeredFaces = facesStacked;
for i = 1: size(facesStacked,2)
    centeredFaces(:,i) = facesStacked(:,i) - meanFace;
end  
meanFace;
centeredFaces;

%% 6. Make Covariance Matrix and Calculating Eigenfaces
A = centeredFaces;
M = A'*A;
[Ui,eigValues,~]= svd(A, 'econ');
figure
D = diag(eigValues);
plot(D)
for i = 1: size(test_images,3)
    Ui(:,i) = Ui(:,i)./ norm(Ui(:,i));
end
Ui;
testU = reshape(Ui(:,2),size(test_images_mouth,1), size(test_images_mouth,2)); % the largest eigenvalue's corresponding eigenface normalized
% imagesc(testU); colormap('gray');

%% 7. Select top  most significant faces
topX = 50;
K = Ui(:,1:topX);
toc 
tic
%% 8. Find a Random Face and guess which face it us
idx = randi(size(faces_test_hard,3));
reshRandomMouth = faces_test_hard(150:225,50:200, idx);

%reshape image to column vector
randomMouth = reshape(faces_test_hard(150:225,50:200,idx),size(faces_test_easy(150:225,50:200,idx),1)*size(faces_test_easy(150:225,50:200,idx),2),1);
%center data on mean
centeredRandomFace = randomMouth - meanFace;
randomMouth;

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

Guess = reshape(facesStacked(:,minIndex),size(T,1), size(T,2)); % the largest eigenvalue's corresponding eigenface normalized
% imagesc(Guess); colormap('gray');

nameofRandom = names_test_hard(:,idx)';
guessName = names_train(:,minIndex)';
toc
%% 9. PLOT ALL THE FACES
subplot(2,1,1), imagesc(reshRandomMouth); colormap('gray');
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
