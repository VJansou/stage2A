close all
clear all
clc
%%
% Dans cette version, on suppose que :
%   - On prend l'objet au moins 3 fois pour chaque angle de vue
%   - Les images sont bien ordonnées
%   - Il n'y a que des images qui nous intéressent dans le dossier
%   - Toutes les images ont le même nombre de pixels
%   - On peut passer l'image en niveau de gris sans problème
%   - On connait le nombre de vues différentes
%%

%% Récupération du jeu de données
folderPath = 'test_00';
nbVues = 5;

fileList = dir(folderPath);
fileList = fileList(3:end); % On enlève .. et .

%% Lire la première image pour déterminer la taille de I et de Slambda
imageRGB = imread(fullfile(folderPath, fileList(1).name));
imageGray = rgb2gray(imageRGB);
[nl, nc] = size(imageGray);
% On réduit les dimensions de l'image
imageGray = imageGray(3*nl/5 - nl/14:3*nl/5 + nl/14,nc/2 - nc/14:nc/2 + nc/14);
imshow(imageGray)
[nnl, nnc] = size(imageGray);
I = zeros(nnl*nnc,length(fileList));
Slambda = zeros(length(fileList),1);

%TEST = imageGray;
%% Boucle principale
for i = 1:length(fileList)
    % Récupération de l'image
    filename = fileList(i).name;
    imageRGB = imread(fullfile(folderPath, filename));
    %imshow(imageRGB);

    % Conversion de l'image en niveau de gris
    imageGray = rgb2gray(imageRGB);
    imageGray = imageGray(3*nl/5 - nl/14:3*nl/5 + nl/14,nc/2 - nc/14:nc/2 + nc/14);
    %TEST = TEST + imageGray;
    %figure(i)
    %subplot(2,1,1)
    %imshow(TEST);
    %subplot(2,1,2)
    %imcontour(TEST,1);
    
    % Ajout dans la colonne i de I le vecteur correspondant aux pixels
    I(:,i) = imageGray(:);

    % Calcul des valeurs singulières de la matrice I(:,1:i) et stockage dans Slambda
    SVD = svd(I(:,1:i));
    if i == 15
        figure
        plot(log(SVD),'x');
    end
    Slambda(i) = sum(SVD);
    
    %% autre sol ?
    %Slambda(i) = sum(svd(double(imageGray)));

    %% Encore autre sol ?
    %Slambda(i) = sum(svd(sum(I,2)));
end
%% Affichage de Slambda
separationTh_00 = [16,27,33,41]; % Numéro des images correspondant au changement d'axe de caméra pour test_00
%separationTh_01 = [6,10,16]; % Numéro des images correspondant au changement d'axe de caméra pour test_01
figure
hold on
plot(Slambda,'r+');
plot(Slambda,'k');
for i = 1:length(separationTh_00)
    xline(separationTh_00(i),'--b');
end
xlabel("Nombre d'image prise en compte")
ylabel("Somme des valeurs singulières de I")
title("Graphique de Slambda")
legend('Valeurs observées', 'Courbe des valeurs observées', 'Changements réels')
hold off

%% Utilisation d'un Slambda relatif
SlambdaRel = zeros(length(Slambda),1);
SlambdaRel(1) = 1; % Pour la première valeur, on la met à 1 pour la comparaison

for i = 2:length(Slambda)
    SlambdaRel(i) = abs(Slambda(i) - Slambda(i-1)) / Slambda(i-1);
end
%% Affichage de SlambdaRel
figure
hold on
plot(SlambdaRel,'r+');
plot(SlambdaRel,'k');
for i = 1:length(separationTh_00)
    xline(separationTh_00(i),'--b');
end
xlabel("Nombre d'image prise en compte")
ylabel("Somme des valeurs singulières de I relatives")
title("Graphique de SlambdaRel")
legend('Valeurs observées', 'Courbe des valeurs observées', 'Changements réels')
hold off

%% Détection changement vue
VariationSLR = zeros(length(SlambdaRel),1);
VariationSLR(1) = 1;
for i = 2:length(VariationSLR)
    VariationSLR(i) = SlambdaRel(i)/ SlambdaRel(i-1);
end
figure
plot(VariationSLR)
findpeaks(VariationSLR)

[sortedValues, sortedIndices] = sort(VariationSLR, 'descend');

% Prendre les indices des 'Vues-1' plus grandes valeurs
indicesChangementVue = sort(sortedIndices(1:(nbVues-1)),'ascend');

%% Création des différents dossiers avec les images
for i = 1:nbVues
    folderName = fullfile('resultat00',sprintf('dossier%d', i));
    mkdir(folderName);

    % Déplacer les images correspondant à ce dossier
    if i == 1
        debut = 1;
    else
        debut = indicesChangementVue(i-1);
    end
    if i == nbVues
        fin = length(fileList);
    else
        fin = indicesChangementVue(i) - 1;
    end
    for j = debut:fin
        source = fullfile(folderPath, fileList(j).name);
        destination = fullfile(folderName, fileList(j).name);
        copyfile(source, destination);
    end
end