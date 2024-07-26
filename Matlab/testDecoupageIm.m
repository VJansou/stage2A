close all
clear all
clc

%% Récupération du jeu de données
folderPath = 'test_00';
nbVues = 5;

fileList = dir(folderPath);
fileList = fileList(3:end); % On enlève .. et .

%% Test avec la première image
imageRGB = imread(fullfile(folderPath, fileList(1).name));
imageGray = rgb2gray(imageRGB);
imageDecoup = Decoupage(imageGray, 0.3);
figure
imshow(imageGray)
figure
imshow(imageDecoup)