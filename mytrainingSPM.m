function [ FOREST C BOW_matrix_cars BOW_matrix_faces ] = mytrainingSPM(k)
% MYTRAINING 
% implementation of BoW algorithm for classification task on CARs and FACEs
%
% inputs:
%   k: parameter k for k-means clustering
%
% outputs:
%   FOREST: k-d tree with k-means centers
%   C: k-means centeros
%   BOW_matrix_cars: histgram of CARs
%   BOW_matrix_faces: histgram of FACEs
%
% Note,
% (1) make sure you have /cars and /faces dataset under this folder
% (2) make sure you add subfolder into addpath
% (3) install vlfeat by yourself or use the (provided) previous version of vlfeat(http://www.vlfeat.org/overview/sift.html)
% (4) you may use your own clustering, or use k-means provided by vlfeat lib
% (5) all you need to modify is in TODOs, please search for TODOs

tic
%% setup dataset and helper functions
addpath('./scripts');
addpath('./vlfeat/toolbox/misc');

run('vlfeat/toolbox/vl_setup');
%% building clusters of words
% extract and collect the sift feature of each images
addpath('./cars'); files_car = dir(['./cars' '/*.jpg']);
addpath('./faces'); files_faces = dir(['./faces' '/*.jpg']);
feature_matrix = [];
% TODOs:
% use first 40 images in both /cars and /faces as training images
% transform each RGB image to gray image, and extract sift feature by vlfeat
% collect all features in 'feature_matrix'(128 by number of all features)
for n = 1:40
    im_car = single(vl_imreadgray(files_car(n).name));
    [f,d] = vl_sift(im_car);
    feature_matrix = [feature_matrix d];
end

for n = 1:40
    im_face = single(vl_imreadgray(files_faces(n).name));
    [f,d] = vl_sift(im_face);
    feature_matrix = [feature_matrix d];
end
% find the centers of features by k-means
feature_matrix = single(feature_matrix);
[C, A] = vl_kmeans(feature_matrix, k);

% compute codewords by kd-tree(vl_kdtreebuild)
% TODOs:
% compute a kd-tree using vlfeat libariry using C above, and output to
% FOREST variable (should be just one line of code)

FOREST = vl_kdtreebuild(C);

%% building bag-of-words for CARs and FACEs
% compute the histogram(frequency) of each training image
addpath('./cars'); files_car = dir(['./cars' '/*.jpg']);
addpath('./faces'); files_faces = dir(['./faces' '/*.jpg']);

BOW_matrix_cars = [];
BOW_matrix_faces = [];
% TODOs:
% now you have centers (codewords) 
%
% you can start to compute the histogram for each training image (/cars, and /faces)
% (1) First, for each image you extract sift descriptors
% (2) then for each extracted descriptor, use kd-tree above to query for index
% (3) build the histogram with all descriptor for one image (remember to normalize it)
% (4) collect all histogram in BOW_matrix_cars and BOW_matrix_faces
%
% hint: BOW_matrix_cars and BOW_matrix_faces are both k by number of images
for i = 1:40
    hist = zeros(1,k);
    image = single(rgb2gray(imread(['cars/' files_car(i).name])));
    [~, car_features] = vl_sift(image);
    [index, ~] = vl_kdtreequery(FOREST, C, single(car_features));
    for j = 1:length(index)
        hist(index(j)) = hist(index(j)) + 1;
    end
    hist = hist/norm(hist);
    %SPM
    histSPM = zeros(4,k);
    [~, carFeat1] = vl_sift(image(1:floor(end/2),1:floor(end/2)));
    [~, carFeat2] = vl_sift(image(1:end/2,floor(end/2)+1:end));
    [~, carFeat3] = vl_sift(image(floor(end/2+1):end,1:floor(end/2)));
    [~, carFeat4] = vl_sift(image(floor(end/2)+1:end,floor(end/2)+1:end));
    
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
    BOW_matrix_cars = [BOW_matrix_cars; [0.5 * pyramid 0.5 * hist]];
end
for i = 1:40
    hist = zeros(1,k);
    image = single(rgb2gray(imread(files_faces(i).name)));
    [~, face_features] = vl_sift(image);
    [index, ~] = vl_kdtreequery(FOREST, C, single(face_features));
    for j = 1:length(index)
        hist(index(j)) = hist(index(j)) + 1;
    end
    hist = hist/norm(hist);
    %spm
    histSPM = zeros(4,k);
    [~, faceFeat1] = vl_sift(image(1:floor(end/2),1:floor(end/2)));
    [~, faceFeat2] = vl_sift(image(1:floor(end/2),floor(end/2):end));
    [~, faceFeat3] = vl_sift(image(floor(end/2):end,1:floor(end/2)));
    [~, faceFeat4] = vl_sift(image(floor(end/2):end,floor(end/2):end));
    
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
    
    BOW_matrix_faces = [BOW_matrix_faces; [0.5 * pyramid 0.5 * hist]];
end
toc









