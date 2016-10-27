function [bwImg,nb,nw] = pixel_extract(image_name)
    oImg = imread(image_name);
    bwImg = im2bw(oImg);
    nb = sum(sum(bwImg==0));
    nw = sum(sum(bwImg==1));
end