function [ correct_car correct_face correctness] = mytestingSPM(  FOREST, C, BOW_matrix_cars, BOW_matrix_faces,k ) 
tic
%MYTESTING Summary of this function goes here
%   Detailed explanation goes here
% inputs
%   FOREST: k-d tree obtained from mytraining.m
%   C: k-means center obtained from mytraining.m
%   BOW_matrix_cars: BOW from mytraining.m
%   BOW_matrix_faces: BOW from mytraining.m
%   k: the same cluster number as mytraining.m
% 
%  outputs
%   correct_car: number of detection
%   correct_face: number of detection
%   correctness: overall recall

addpath('./scripts');
addpath('./vlfeat/toolbox/misc');
run('vlfeat/toolbox/vl_setup');

correct_car = 0;
correct_face = 0;

addpath('./cars'); files = dir(['./cars' '/*.jpg']);
for i=41:90
    disp(files(i).name); 
    I = single(rgb2gray(imread(files(i).name)));
    % TODOs:
    % for each of testing image, 
    % (1) extract sift descriptor as in training process
    % (2) compute histogram
    % (3) store normalized histogram as variable v for knnsearch
    % (4) assign it to the closer object
    [~, car_features] = vl_sift(I);
    [index, ~] = vl_kdtreequery(FOREST, C, single(car_features));
    hist = zeros(1,k);
    for j = 1:length(index)
        hist(index(j)) = hist(index(j)) + 1;
    end
    v = hist/norm(hist); 
    %spm
    histSPM = zeros(4,k);
    [~, carFeat1] = vl_sift(I(1:floor(end/2),1:floor(end/2)));
    [~, carFeat2] = vl_sift(I(1:floor(end/2),floor(end/2)+1:end));
    [~, carFeat3] = vl_sift(I(floor(end/2)+1:end,1:floor(end/2)));
    [~, carFeat4] = vl_sift(I(floor(end/2)+1:end,floor(end/2)+1:end));
    
    [index1, ~] = vl_kdtreequery(FOREST, C, single(carFeat1));
    [index2, ~] = vl_kdtreequery(FOREST, C, single(carFeat2));
    [index3, ~] = vl_kdtreequery(FOREST, C, single(carFeat3));
    [index4, ~] = vl_kdtreequery(FOREST, C, single(carFeat4));
    
    for j = 1:length(index1)
        histSPM(1,index1(j)) = histSPM(1,index1(j)) + 1;
    end    
    for j = 1:length(index2)
        histSPM(2,index2(j)) = histSPM(2,index2(j)) + 1;
    end
    for j = 1:length(index3)
        histSPM(3,index3(j)) = histSPM(3,index3(j)) + 1;
    end
    for j = 1:length(index4)
        histSPM(4,index4(j)) = histSPM(4,index4(j)) + 1;
    end
    
%     histSPM(1,:) = histSPM(1,:)./norm(histSPM(1,:));
%     histSPM(2,:) = histSPM(2,:)./norm(histSPM(2,:));
%     histSPM(3,:) = histSPM(3,:)./norm(histSPM(3,:));
%     histSPM(4,:) = histSPM(4,:)./norm(histSPM(4,:));
    pyramid = [histSPM(1,:) histSPM(2,:) histSPM(3,:) histSPM(4,:)];
    pyramid = pyramid./sum(pyramid);
    v = [0.5 * pyramid 0.5 * v];
    
    
    [IDX d_car] = knnsearch(BOW_matrix_cars,v);
    [IDX d_face] = knnsearch(BOW_matrix_faces,v);
    if(d_car < d_face)
        correct_car = correct_car+1;
    end
end
clear files

addpath('./faces'); files = dir(['./faces' '/*.jpg']);
for i=41:90
    disp(files(i).name); 
    I = single(rgb2gray(imread(files(i).name)));
    % TODOs:
    % same as above except this is for face
    [~, face_features] = vl_sift(I);
    [index, ~] = vl_kdtreequery(FOREST, C, single(face_features));
    hist = zeros(1,k);
    for j = 1:length(index)
        hist(index(j)) = hist(index(j)) + 1;
    end
    v = hist/norm(hist);
    %SPM
    histSPM = zeros(4,k);
    [~, faceFeat1] = vl_sift(I(1:floor(end/2),1:floor(end/2)));
    [~, faceFeat2] = vl_sift(I(1:floor(end/2),floor(end/2):end));
    [~, faceFeat3] = vl_sift(I(floor(end/2):end,1:floor(end/2)));
    [~, faceFeat4] = vl_sift(I(floor(end/2):end,floor(end/2):end));
    
    [index1, ~] = vl_kdtreequery(FOREST, C, single(faceFeat1));
    [index2, ~] = vl_kdtreequery(FOREST, C, single(faceFeat2));
    [index3, ~] = vl_kdtreequery(FOREST, C, single(faceFeat3));
    [index4, ~] = vl_kdtreequery(FOREST, C, single(faceFeat4));
    
    for j = 1:length(index1)
        histSPM(1,index1(j)) = histSPM(1,index1(j)) + 1;
    end    
    for j = 1:length(index2)
        histSPM(2,index2(j)) = histSPM(2,index2(j)) + 1;
    end
    for j = 1:length(index3)
        histSPM(3,index3(j)) = histSPM(3,index3(j)) + 1;
    end
    for j = 1:length(index4)
        histSPM(4,index4(j)) = histSPM(4,index4(j)) + 1;
    end
    
%     histSPM(1,:) = histSPM(1,:)./norm(histSPM(1,:));
%     histSPM(2,:) = histSPM(2,:)./norm(histSPM(2,:));
%     histSPM(3,:) = histSPM(3,:)./norm(histSPM(3,:));
%     histSPM(4,:) = histSPM(4,:)./norm(histSPM(4,:));
    
    pyramid = [histSPM(1,:) histSPM(2,:) histSPM(3,:) histSPM(4,:)];
    pyramid = pyramid./sum(pyramid);
    v = [0.5 * pyramid 0.5 * v];
    
    [IDX d_car] = knnsearch(BOW_matrix_cars,v);
    [IDX d_face] = knnsearch(BOW_matrix_faces,v);
    if(d_face < d_car)
        correct_face = correct_face+1;
    end
end
clear files

correctness = (correct_car+correct_face)/100;
toc

