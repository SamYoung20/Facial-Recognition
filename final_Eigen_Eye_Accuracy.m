function x = final_Eigen_Eye_Accuracy( )
%% 1. Load in Test Data

load face_detect.mat
faces_test_easy = faces_test_easy;
faces_test_hard = faces_test_hard;
faces_train = faces_train;
names_test_easy = names_test_easy;
names_test_hard = names_test_hard;
names_train = names_train;
test_images = faces_train;
test_images_eyes= test_images(70:120,50:230,:);
sizes = size(faces_test_hard,3)

%% 2. Reshape into column vectors for each image.

test_images_eyes= test_images(70:120,50:230,:);
T = test_images(70:120,50:230, 1);
facesStacked = reshape(test_images_eyes ,size(test_images_eyes,1)*size(test_images_eyes,2),size(test_images_eyes,3));

%% 3. Find Mean Face

sumFace = [];
for i = 1: size(facesStacked,1)
    sumFace(i,1) = sum(facesStacked(i,:));
end    
meanFace =sumFace./size(test_images_eyes,3);

%% 4. Test Mean Face



%% 5. Recenter Faces vs. mean Face

centeredFaces = facesStacked;
for i = 1: size(facesStacked,2)
    centeredFaces(:,i) = facesStacked(:,i) - meanFace;
end  

%% 6. Make Covariance Matrix and Calculating Eigenfaces

A = centeredFaces;
M = A'*A;
[Ui,eigValues,~]= svd(A, 'econ');

%Ui are the eigen faces
for i = 1: size(test_images,3)
    Ui(:,i) = Ui(:,i)./ norm(Ui(:,i));
end

%% 7. Select top  most significant faces

topX = 170;
K = Ui(:,1:topX);

%% accuracy
%%MAIN
counter = 0;
for i = 1:sizes
    [face1, face2] = guessWhichFace(i, meanFace, topX, K, centeredFaces,facesStacked, counter,  names_test_hard, faces_test_hard);
    if face1 == face2
        counter= counter + 1;
    end
end
counter1 = 0;
for i = 1:sizes
    [face1, face2] = guessWhichFace(i, meanFace, topX, K, centeredFaces,facesStacked, counter1,  names_test_easy, faces_test_easy);
    if face1 == face2
        counter1 = counter1 + 1;
    end
end
accuracyHard = counter/size(faces_test_hard,(3))
accuracyEasy = counter1/size(faces_test_easy,(3))

%% 8. Find a Random Face and guess which face it is
    function [face1,face2] = guessWhichFace(index,meanFace, topX, K, centeredFaces,facesStacked,counter, names_test1, faces_test1)
        idx = index;
        names_test = names_test1;
        faces_test = faces_test1;
        nameofRandom = names_test(:,idx)';
        randomOriginal = faces_test(70:120,50:230,idx);
        reshRandom = faces_test(70:120,50:230, idx);

        %reshape image to column vector
        randomEyes = reshape(faces_test_hard(70:120,50:230,idx),size(faces_test_easy(70:120,50:230,idx),1)*size(faces_test_easy(70:120,50:230,idx),2),1);
        %center data on mean
        centeredRandomFace = randomEyes - meanFace;
        weightsRando = [];
        for i =1:topX
            weightsRando(i,1) = K(:,i)'*centeredRandomFace;
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
        nameofRandom = names_test(:,idx)';
        guessName = names_train(:,minIndex)';
        face1 = guessName;
        face2 = nameofRandom;
    end

end

