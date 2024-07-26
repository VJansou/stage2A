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

%% Choix du jeu de données
folderPath = 'test_03';
PourcentageImage = 0.4;
out = 'resultat_03';
% separationTh = [16,27,33,41]; % Numéro des images correspondant au changement d'axe de caméra pour test_00
%separationTh = [6,10,16]; % Numéro des images correspondant au changement d'axe de caméra pour test_01
separationTh = [11,21,31,41,51,61,71,81,91]; % Numéro des images correspondant au changement d'axe de caméra pour test_03
nbVues = length(separationTh) + 1;
%% Récupération du jeu de données
fileList = dir(folderPath);
fileList = fileList(3:end); % On enlève .. et .

%% Lire la première image pour déterminer la taille de I et de Slambda
imageRGB = imread(fullfile(folderPath, fileList(1).name));
imageGray = rgb2gray(imageRGB);
% On réduit les dimensions de l'image
%imageGray = Decoupage(imageGray, PourcentageImage);
[nl, nc] = size(imageGray);
%imshow(imageGray)
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
    imageGray = MasqueV2(imageGray);
    %imageGray = Decoupage(imageGray, PourcentageImage);
    %figure
    %imshow(imageGray)
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
        plot(log(SVD),'+');
    end
    Slambda(i) = sum(SVD);
    
    %% autre sol ?
    %Slambda(i) = sum(svd(double(imageGray)));

    %% Encore autre sol ?
    %Slambda(i) = sum(svd(sum(I,2)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Analyse %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Affichage de Slambda
figure
hold on
plot(Slambda,'r+');
plot(Slambda,'k');
for i = 1:length(separationTh)
    xline(separationTh(i),'--b');
end
xlabel("Nombre d'images prisent en compte")
ylabel("Somme des valeurs singulières de I")
title("Graphique de Slambda")
legend('Valeurs observées', 'Courbe des valeurs observées', 'Changements réels')
hold off

%% On dérive pour analyser les variations
SlambdaDer = gradient(Slambda,linspace(1,length(fileList),length(fileList)));
SlambdaDer2 = gradient(SlambdaDer,linspace(1,length(fileList),length(fileList)));
figure
subplot(3,1,1)
plot(Slambda)
title("Somme des valeurs singulières en fonction du nombre d'image")
subplot(3,1,2)
plot(SlambdaDer)
title("Dérivée de Slambda")
subplot(3,1,3)
plot(SlambdaDer2)
title("Dérivée seconde de Slambda")


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
for i = 1:length(separationTh)
    xline(separationTh(i),'--b');
end
xlabel("Nombre d'image prisent en compte")
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
xlabel("Nombre d'images prisent en compte")
ylabel("SlambdaRel(i)/ SlambdaRel(i-1)");
title('Variations de SlambdaRel')

[sortedValues, sortedIndices] = sort(VariationSLR, 'descend');

% Prendre les indices des 'Vues-1' plus grandes valeurs
indicesChangementVue = sort(sortedIndices(1:(nbVues-1)),'ascend');

%% Création des différents dossiers avec les images
for i = 1:nbVues
    folderName = fullfile(out,sprintf('dossier%d', i));
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