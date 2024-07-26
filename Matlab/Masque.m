function [Imageoutput] = Masque(Imageinput)
%MASQUE prend une image en niveaux de gris et retourne un masque binaire de
%l'image
    %% Détection de l'objet
    [~,threshold] = edge(Imageinput,'sobel');
    fudgeFactor = 0.3;
    BWs = edge(Imageinput,'sobel',threshold * fudgeFactor);
    
    %% Dilatation de l'image
    se90 = strel('line',3,90);
    se0 = strel('line',3,0);
    BWsdil = imdilate(BWs,[se90 se0]);
    
    %% Remplissage de l'image
    BWdfill = imfill(BWsdil,'holes');
    
    %% Isolation de l'objet
    BWnobord = imclearborder(BWdfill,26);
    
    %% Lissage de l'objet
    seD = strel('disk',5);
    BWfinal = imopen(BWnobord,seD);
    %BWfinal = imclose(BWfinal,seD);
    Imageoutput = bit2int(BWfinal,1);
end

