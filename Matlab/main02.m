close all
clear all
clc
%%
% Dans cette version, on suppose que :
%   - On prend l'objet au moins 3 fois pour chaque angle de vue
%   - Les images sont bien ordonnées
%   - Il n'y a que des images qui nous intéressent dans le dossier
%   - Toutes les images ont le même nombre de pixels
%   - On connait le nombre de vues différentes
%
% L'idée est ici d'étudier l'efficacité d'une méthode étudiant les 3 canaux
% RGB
%%

%% Paramétrage du jeu de données
folderPath = 'test_02'; % dossier du jeu de données
typeImage = '.png'; % type des images : réduit déjà la possibilité de trier une autre image par erreur
pourcentageImageGardee = 0.4; % on réduit les dimensions de l'image pour atténuer l'effet de l'arrière-plan
out = 'resulta_02'; % dossier dans lequel sera rangé le jeu de données trié

% separationTh = [16,27,33,41]; % Numéro des images correspondant au changement d'axe de caméra pour test_00
separationTh = [6,10,16]; % Numéro des images correspondant au changement d'axe de caméra pour test_01
% separationTh = [11,21,31,41,51,61,71,81,91]; % Numéro des images correspondant au changement d'axe de caméra pour test_03
nbVues = length(separationTh) + 1;

%% Récupération du jeu de données
fileList = dir(folderPath);
fileList = fileList(endsWith({fileList.name},typeImage,'IgnoreCase',true)); % On filtre les fichiers
nbImage = length(fileList);

%% Initialisation matrice I et Slambda
imageRGB = imread(fullfile(folderPath, fileList(1).name));
imageRGB = Decoupage(rgb2gray(imageRGB), pourcentageImageGardee);
[nl, nc, ~] = size(imageRGB);
I = zeros(nl*nc, nbImage, 3);
Slambda = zeros(nbImage,3);

%% Remplissage de I et calcul des SVD
for i = 1:nbImage
    % Récupération et préparation de l'image
    filename = fileList(i).name;
    imageRGB = imread(fullfile(folderPath, filename));
    % On sépare pour chaque canal
    for canal = 1:3
        image = Decoupage(imageRGB(:,:,canal), pourcentageImageGardee);
        I(:,i,canal) = image(:);
        SVD = svd(I(:,1:i,canal));
        Slambda(i,canal) = sum(SVD);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Analyse %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lp = linspace(1,nbImage,nbImage);
SlambdaDer = zeros(nbImage,3);
SlambdaDer2 = zeros(nbImage,3);
SlambdaRel = zeros(nbImage,3);

for canal = 1:3
    SlambdaDer(:,canal) = gradient(Slambda(:,canal), lp);
    SlambdaDer2(:,canal) = gradient(SlambdaDer(:,canal), lp);
    SlambdaRel(1,canal) = 1;
    for i = 2:nbImage
        SlambdaRel(i,canal) = abs(Slambda(i,canal) - Slambda(i-1,canal)) / Slambda(i-1,canal);
    end
end
VariationSLR = zeros(nbImage,3);
for canal = 1:3
    VariationSLR(1,canal) = 1;
    for i = 2:nbImage
        VariationSLR(i,canal) = SlambdaRel(i,canal)/ SlambdaRel(i-1,canal);
    end
end

%% Affichage des courbes
legende = {'Canal Rouge', 'Canal Vert', 'Canal Bleu'};

% Slambda
figure
hold on
for canal = 1:3
    plot(Slambda(:,canal))
end
xline(separationTh,'--k');
legend(legende);
title("Somme des valeurs singulières en fonction du nombre d'image")
hold off

% SlambdaDer
figure
hold on
for canal = 1:3
    plot(SlambdaDer(:,canal))
end
xline(separationTh,'--k');
legend(legende);
title("Dérivée de Slambda")
hold off

% SlambdaDer2
figure
hold on
for canal = 1:3
    plot(SlambdaDer2(:,canal))
end
xline(separationTh,'--k');
legend(legende);
title("Dérivée seconde de Slambda")
hold off

% SlambdaRel
figure
hold on
for canal = 1:3
    plot(VariationSLR(:,canal))
end
xline(separationTh,'--k');
legend(legende);
title("Variations de SlambdaRel")
hold off

%% Indices pris séparément

% Prendre les indices des 'Vues-1' plus grandes valeurs
indicesChangementVue = zeros(nbVues - 1,3);
for canal = 1:3
    [sortedValues, sortedIndices] = sort(VariationSLR(:,canal), 'descend');
    indicesChangementVue(:,canal) = sort(sortedIndices(1:(nbVues-1)),'ascend');
end
indicesChangementVue
%% Regroupement des résultats des 3 canaux