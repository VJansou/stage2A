function [Imageoutput] = Decoupage(Imageinput, factor)
%DECOUPAGE renvoie une partie de l'image

% Vérifie que le facteur est compris entre 0 et 1
    if factor <= 0 || factor > 1
        error('Le facteur doit être un nombre entre 0 (exclus) et 1 (inclus).');
    end

    % Obtenir les dimensions de l'image originale
    [height, width] = size(Imageinput);

    % Calculer les nouvelles dimensions
    newHeight = round(height * factor);
    newWidth = round(width * factor);

    % Calculer les coordonnées de recadrage
    startY = round((height - newHeight) / 2) + 1;
    startX = round((width - newWidth) / 2) + 1;
    endY = startY + newHeight - 1;
    endX = startX + newWidth - 1;

    % Recadrer l'image
    Imageoutput = Imageinput(startY:endY, startX:endX, :);
end

