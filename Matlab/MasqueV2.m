function [Imageoutput] = MasqueV2(Imageinput)
%MASQUE prend une image en niveaux de gris et retourne un masque binaire de
%l'image
     %% Détection de l'objet
    [~,threshold] = edge(Imageinput,'sobel');
    fudgeFactor = 0.3;
    BWs = edge(Imageinput,'sobel',threshold * fudgeFactor);

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

    % figure
    % subplot(1,5,1)
    % imshow(BWs);
    % subplot(1,5,2)
    % imshow(BWnobord)
    % subplot(1,5,3)
    % imshow(BWsdil)
    % subplot(1,5,4)
    % imshow(BWdfill)
    % subplot(1,5,5)
    % imshow(BWfinal)
end

