function [pad_bwImg] = missing_data(image_name,pad)
    [bwImg,nb,nw] = pixel_extract(image_name);
    %disp(bwImg);
    rsize = size(bwImg,1);
    pad_bwImg = [bwImg(1:pad,:);bwImg;bwImg(rsize:-1:rsize-pad+1,:)];
    pcsize = size(pad_bwImg,2);
    pad_bwImg = [pad_bwImg(:,1:pad), pad_bwImg, ...
        pad_bwImg(:,pcsize:-1:pcsize-pad+1)];
end