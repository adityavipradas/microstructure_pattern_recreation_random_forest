function[] = train_data_causal_VF(image_name,pad,h,w)
    clc;
    [bwImg,nb,nw] = pixel_extract(image_name);
    
    figure();
    imshow(bwImg);
    title('MATLAB converted image');
    
    strb = sprintf('Original Image Black Pixels: %d', nb); 
    strw = sprintf('Original Image White Pixels: %d', nw);
    stres = sprintf('Image Resolution %d x %d', size(bwImg,1),size(bwImg,2));
    %disp(stres);
    disp(strb);
    disp(strw);
    
    [pad_bwImg] = missing_data(image_name,pad);
    
    figure();
    imshow(pad_bwImg);
    title('Boundary padding');
    
    prsize = size(pad_bwImg,1);
    pcsize = size(pad_bwImg,2);
    rows = pad-h:prsize-pad;
    cols = pad-w:pcsize-(pad-w);
    Tmat = pad_bwImg(rows, cols);
    Y = reshape(Tmat',numel(Tmat),1);
    X = zeros(numel(Tmat),h*(2*w+1)+w);
    r = 1;
    for i = rows
        for j = cols
            c = 1;
            for k = i-h:1:i-1
                for l = j-w:1:j+w
                    X(r,c) = pad_bwImg(k,l);
                    c = c + 1;
                end
            end
            for l = j-w:1:j-1
                X(r,c) = pad_bwImg(i,l);
                c = c + 1;
            end
            r = r + 1;
        end
    end
    btree = fitctree(X,Y);
    %view(btree);
    
    
    %CREATE INITIAL IMAGES
    
    %white
    %RImg = ones(size(pad_bwImg));
    
    %black
    RImg = zeros(size(pad_bwImg));
    
    %random
    %RImg = randi([0,1],size(pad_bwImg));
    
    %splices
    %RImg = zeros(size(pad_bwImg));
    %initR = floor(size(pad_bwImg,1)/pad);
    %RImg_temp = repmat(pad_bwImg(1:pad,:), initR, 1);
    %RImg(1:initR*pad,:) = RImg_temp;
    
    %original image
    %RImg = pad_bwImg;
    
    figure();
    imshow(RImg);
    title('Initial image');
    
    newX = zeros(1, h*(2*w+1)+w);
    %disp(rows);
    nb_new = 0;
    nw_new = 0;
    %Bernoulli adjustment parameter
    para = 0;
    iter = 0;
    while abs(nb - nb_new) > 500
        iter = iter + 1;
        for i = rows
            for j = cols
                c = 1;
                for k = i-h:1:i-1
                    for l = j-w:1:j+w
                        newX(1,c) = RImg(k,l);
                        c = c + 1;
                    end
                end
                for l = j-w:1:j-1
                    newX(1,c) = RImg(i,l);
                    c = c + 1;
                end
                [label, score] = predict(btree,newX);
                threshold = score(1) + para*sqrt(score(1)*score(2));
                %inverse transform method
                a = rand(1);
                if a < threshold
                    RImg(i,j) = 0;
                else
                    RImg(i,j) = 1;
                end
            end
        end
        nb_new = sum(sum(RImg(pad+1:prsize-pad,pad+1:pcsize-pad)==0));
        nw_new = sum(sum(RImg(pad+1:prsize-pad,pad+1:pcsize-pad)==1));
        strb = sprintf('Reconstructed Image %d Black Pixels: %d', iter,nb_new); 
        strw = sprintf('Reconstructed Image %d White Pixels: %d', iter,nw_new);
        disp(strb);
        disp(strw);
        
        if nb > nb_new
            para = 0.005;
        elseif nb < nb_new
            para = -0.005;
        end
        figure();
        imshow(RImg(pad+1:prsize-pad,pad+1:pcsize-pad));
        str = sprintf('Reconstructed image %d',iter);
        title(str);
    end
end