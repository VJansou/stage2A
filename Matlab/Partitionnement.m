function [pieces] = Partitionnement(image, nbRows, nbCols)
% Partitionnement - Divise une image en 6 morceaux rectangulaires
% Input:
%   image - image à partitionner
%   nbRows - Nombre de lignes pour le partitionnement
%   nbCols - Nombre de colonnes pour le partitionnement
% Output:
%   pieces - Cellule contenant les morceaux de l'image

    % Obtenir les dimensions de l'image
    [height, width] = size(image);
    
    % Calculer les dimensions de chaque morceau
    pieceHeight = floor(height / nbRows);
    pieceWidth = floor(width / nbCols);
    
    % Initialiser la cellule pour contenir les morceaux
    pieces = cell(nbRows, nbCols);

    % Extraire chaque morceau
    for row = 1:nbRows
        for col = 1:nbCols
            % Définir les limites de chaque morceau
            rowStart = (row-1) * pieceHeight + 1;
            rowEnd = min(row * pieceHeight, height);
            
            colStart = (col-1) * pieceWidth + 1;
            colEnd = min(col * pieceWidth, width);
            
            % Extraire le morceau
            pieces{row, col} = image(rowStart:rowEnd, colStart:colEnd, :);
        end
    end
end