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
%
% L'idée est ici d'étudier l'efficacité d'une méthode visant à diviser
% l'image en plusieurs morceaux
%%

%% Paramétrage du jeu de données
folderPath = 'test_03'; % dossier du jeu de données
typeImage = '.png'; % type des images : réduit déjà la possibilité de trier une autre image par erreur
pourcentageImageGardee = 1; % on réduit les dimensions de l'image pour atténuer l'effet de l'arrière-plan
nbRows = 1; % nombre de lignes pour le partitionnement
nbCols = 1; % nombre de colonnes pour le partitionnement
out = 'resulta_03'; % dossier dans lequel sera rangé le jeu de données trié

% separationTh = [16,27,33,41]; % Numéro des images correspondant au changement d'axe de caméra pour test_00
%separationTh = [6,10,16]; % Numéro des images correspondant au changement d'axe de caméra pour test_01
separationTh = [11,21,31,41,51,61,71,81,91]; % Numéro des images correspondant au changement d'axe de caméra pour test_03
nbVues = length(separationTh) + 1;

%% Récupération du jeu de données
fileList = dir(folderPath);
fileList = fileList(endsWith({fileList.name},typeImage,'IgnoreCase',true)); % On filtre les fichiers
nbImage = length(fileList);

%% Partitionnement d'une image pour visualiser et initialisation des matrices I
imageRGB = imread(fullfile(folderPath, fileList(1).name));
imageGray = rgb2gray(imageRGB);
imageGray = Decoupage(imageGray, pourcentageImageGardee);
pieces = Partitionnement(imageGray,nbRows,nbCols);

% Afficher les morceaux
figure
for row = 1:nbRows
    for col = 1:nbCols
        subplot(nbRows, nbCols, (row-1)*nbCols + col);
        imshow(pieces{row, col});
        title(['Morceau ', num2str((row-1)*nbCols + col)]);
    end
end

% Ajout du masque
% for row = 1:nbRows
%     for col = 1:nbCols
%         pieces{row, col} = MasqueV2(pieces{row, col});
%     end
% end

% Afficher les morceaux
figure
for row = 1:nbRows
    for col = 1:nbCols
        subplot(nbRows, nbCols, (row-1)*nbCols + col);
        imshow(pieces{row, col});
        title(['Morceau ', num2str((row-1)*nbCols + col)]);
    end
end

% Dimensions des partitions
partitionHeights = cellfun(@(x) size(x, 1), pieces);
partitionWidths = cellfun(@(x) size(x, 2), pieces);

% Initialisation des matrices pour chaque partition
I = cell(nbRows, nbCols);
Slambda = cell(nbRows, nbCols);
for row = 1:nbRows
    for col = 1:nbCols
        I{row, col} = zeros(partitionHeights(row, col)*partitionWidths(row, col), nbImage);
        Slambda{row,col} = zeros(nbImage,1);
    end
end

%% Remplissage des matrices I{row, col} et calcul des SVD
for i = 1:nbImage

    % Récupération et préparation de l'image
    filename = fileList(i).name;
    imageRGB = imread(fullfile(folderPath, filename));
    imageGray = rgb2gray(imageRGB);
    imageGray = Decoupage(imageGray, pourcentageImageGardee);       
    pieces = Partitionnement(imageGray,nbRows,nbCols);

    % Ajout des partitions dans les cellules I{row, col}
    for row = 1:nbRows
        for col = 1:nbCols
            img = pieces{row, col};
            % img = MasqueV2(img); % application du masque
            I{row, col}(:,i) = img(:);

            % Calcul des valeurs singulières de la matrice I{row, col}(:,1:i)
            % et stockage dans Slambda{row, col}(i)
            SVD = svd(I{row, col}(:,1:i));
            Slambda{row, col}(i) = sum(SVD);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Analyse %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Calcul des variations de Slambda
SlambdaDer = cell(nbRows, nbCols);
SlambdaDer2 = cell(nbRows, nbCols);
legende = cell (nbRows*nbCols, 1);
idx = 1;
lp = linspace(1,nbImage,nbImage);
for row = 1:nbRows
    for col = 1:nbCols
        SlambdaDer{row, col} = gradient(Slambda{row, col}, lp);
        SlambdaDer2{row, col} = gradient(SlambdaDer{row, col}, lp);
        legende{idx} = sprintf('Morceau %d', (row-1)*nbCols + col);
        idx = idx + 1;
    end
end

%% Affichage de Slambda et de ses dérivées

% Slambda
figure
hold on
for row = 1:nbRows
    for col = 1:nbCols
        plot(Slambda{row, col});
    end
end
xline(separationTh,'--b');
legend(legende);
title("Somme des valeurs singulières en fonction du nombre d'image")
hold off

%SlambdaDer
figure
hold on
for row = 1:nbRows
    for col = 1:nbCols
        plot(SlambdaDer{row, col});
    end
end
xline(separationTh,'--b');
legend(legende);
title("Dérivée de Slambda")
hold off

%SlambdaDer
figure
hold on
for row = 1:nbRows
    for col = 1:nbCols
        plot(SlambdaDer2{row, col});
    end
end
xline(separationTh,'--b');
legend(legende);
title("Dérivée seconde de Slambda")
hold off

%% Pareil avec un Slambda relatif
SlambdaRel = cell(nbRows, nbCols);
VariationSLR = cell(nbRows, nbCols);
for row = 1:nbRows
    for col = 1:nbCols
        SlambdaRel{row, col} = zeros(nbImage,1);
        VariationSLR{row, col} = zeros(nbImage,1);
    end
end
for row = 1:nbRows
    for col = 1:nbCols
        SlambdaRel{row, col}(1) = 1;
        for i = 2:nbImage
            SlambdaRel{row, col}(i) = abs(Slambda{row, col}(i) - Slambda{row, col}(i-1)) / Slambda{row, col}(i-1);
        end
    end
end
for row = 1:nbRows
    for col = 1:nbCols
        VariationSLR{row, col}(1) = 1;
        for i = 2:nbImage
            VariationSLR{row, col}(i) = SlambdaRel{row, col}(i)/ SlambdaRel{row, col}(i-1);
        end
    end
end

% Affichage des courbes
figure
hold on
for row = 1:nbRows
    for col = 1:nbCols
        plot(VariationSLR{row, col});
    end
end
xline(separationTh,'--b');
legend(legende);
title("Variations de SlambdaRel")
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Tests %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% On va voir si faire une moyenne peut être une bonne solution
MoyenneVar = zeros(nbImage,1);

for row = 1:nbRows
    for col = 1:nbCols
        MoyenneVar = MoyenneVar + VariationSLR{row, col};
    end
end
MoyenneVar = MoyenneVar / (nbRows*nbCols);

figure
plot(MoyenneVar)
xline(separationTh,'--b');
title("Moyenne des variations de SlambdaRel")