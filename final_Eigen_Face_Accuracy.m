function x = final_Eigen_Face_Accuracy( )
%% 1. Load in Test Data
tic
load face_detect.mat
faces_test_easy = faces_test_easy;
faces_test_hard = faces_test_hard(:,:,1:40);
faces_train = faces_train;
names_test_easy = names_test_easy;
names_test_hard = names_test_hard;
names_train = names_train;
test_images = faces_train;


%% 2. Reshape into column vectors for each image.

facesStacked = reshape(test_images ,size(test_images,1)*size(test_images,2),size(test_images,3));
T = test_images(:,:, 1);

%% 3. Find Mean Face

sumFace = [];
for i = 1: size(facesStacked,1)
    sumFace(i,1) = sum(facesStacked(i,:));
end    
meanFace =sumFace./size(test_images,3);

%% 4. Test Mean Face



%% 5. Recenter Faces vs. mean Face

centeredFaces = facesStacked;
for i = 1: size(facesStacked,2)
    centeredFaces(:,i) = facesStacked(:,i) - meanFace;
end  

%% 6. Make Covariance Matrix and Calculating Eigenfaces

A = centeredFaces;
M = A'*A;
[Vi,eigValues]= eig(M);
for i = 1: size(test_images,3)
    Ui(:,i) = A*Vi(:,i); 
end

%Ui are the eigen faces
for i = 1: size(test_images,3)
    Ui(:,i) = Ui(:,i)./ norm(Ui(:,i));
end
%% 7. Select top  most significant faces

topX = 30;
K = Ui(:,240-topX:240);
toc
tic
%% accuracy
%%MAIN
counter = 0;
%test all images in hard data set
for i = 1:40
    [face1, face2] = guessWhichFace(i, meanFace, topX, K, centeredFaces,facesStacked, counter,  names_test_hard, faces_test_hard);
    if face1 == face2
        counter= counter + 1;
    end
end
toc
counter1 = 0;
tic
%test all images in easy data set
for i = 1:40
    [face1, face2] = guessWhichFace(i, meanFace, topX, K, centeredFaces,facesStacked, counter1,  names_test_easy, faces_test_easy);
    if face1 == face2 % Checks to see if the predicted face matches teh test face
        counter1 = counter1 + 1;
    end
end
toc
accuracyHard = counter/size(faces_test_hard,(3))
accuracyEasy = counter1/size(faces_test_easy,(3))

%% 8. Find a Random Face and guess which face it is
    function [face1,face2] = guessWhichFace(index,meanFace, topX, K, centeredFaces,facesStacked,counter, names_test1, faces_test1)
        idx = index;
        names_test = names_test1;
        faces_test = faces_test1;
        nameofRandom = names_test(:,idx)';
        randomOriginal = faces_test(:,:,idx);
        reshRandom = faces_test(:,:, idx);

        %reshape image to column vector
        randomFace = reshape(faces_test(:,:,idx),size(faces_test(:,:,idx),1)*size(faces_test(:,:,idx),2),1);
        %center data on mean
        centeredRandomFace = randomFace - meanFace;
        weightsRando = [];
        for i =1:topX
            weightsRando(i,1) = K(:,i)'*centeredRandomFace; %project onto eigenspace - finds weights for linear combination
        end
        weightsTraining = [];
        for i =1:topX
            weightsTraining(i,:) = K(:,i)'*centeredFaces;
        end
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
        imagesc(Guess); colormap('gray');
        nameofRandom = names_test(:,idx)';
        guessName = names_train(:,minIndex)';
        face1 = guessName;
        face2 = nameofRandom;
    end

end

