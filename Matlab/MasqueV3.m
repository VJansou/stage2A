function Imageoutput = MasqueV3(Imageinput)
%MASQUEV3 prend une image en niveaux de gris et retourne un masque binaire de
%l'image
    % %% Détection de l'objet
    % % [~,threshold] = edge(Imageinput,'sobel');
    % % fudgeFactor = 0.3;
    % % BWs = edge(Imageinput,'sobel',threshold * fudgeFactor);
    % BWs = edge(Imageinput,'sobel');
    % figure
    % subplot(1,4,1)
    % imshow(BWs);
    % 
    % %% Remplissage de l'image
    % BWdfill = imfill(BWs,'holes');
    % subplot(1,4,2)
    % imshow(BWs);
    % 
    % %% Lissage de l'objet
    % seD = strel('disk',2);
    % BWfinal = imopen(BWdfill,seD);
    % subplot(1,4,4)
    % imshow(BWfinal);
    % Imageoutput = bit2int(BWfinal,1);
    [Gmag,Gdir] = imgradient(Imageinput,"prewitt");
    figure
    subplot(1,2,1)
    imshowpair(Gmag,Gdir,"montage");
    subplot(1,2,2)
    imshow(Gmag)
    Imageoutput = Gmag;
end

