close all
clear all
clc

%% Récupération du jeu de données
folderPath = 'test_00';
%folderPath = fullfile('Donnees_1','Auguste','103','jpeg-exports');
nbVues = 5;

fileList = dir(folderPath);
fileList = fileList(3:end); % On enlève .. et .

% for i = 1:length(fileList)
%     imageRGB = imread(fullfile(folderPath, fileList(i).name));
%     imageGray = rgb2gray(imageRGB);
% 
%     masque = Masque(imageGray);
%     figure
%     imshow(masque)
% end

%% MASQUEV2
for i = 1:1%length(fileList)
    imageRGB = imread(fullfile(folderPath, fileList(i).name));
    figure
    imshow(imageRGB(:,:,1))
    figure
    imshow(imageRGB(:,:,2))
    figure
    imshow(imageRGB(:,:,3))
    imageGray = rgb2gray(imageRGB);
    %% Détection de l'objet
    [~,threshold] = edge(imageGray,'sobel');
    fudgeFactor = 0.3;
    BWs = edge(imageGray,'sobel',threshold * fudgeFactor);
    
    
    %% Isolation de l'objet
    BWnobord = imclearborder(BWs,4);

    %% Dilatation de l'image
    se90 = strel('line',3,90);
    se0 = strel('line',3,0);
    BWsdil = imdilate(BWnobord,[se90 se0]);
    %% Remplissage de l'image
    BWdfill = imfill(BWsdil,'holes');

    
    %% Lissage de l'objet
    seD = strel('disk',5);
    BWfinal = imopen(BWdfill,seD);
    %BWfinal = imclose(BWfinal,seD);
    Imageoutput = bit2int(BWfinal,1);
    figure
    imshow(Imageoutput)
end

