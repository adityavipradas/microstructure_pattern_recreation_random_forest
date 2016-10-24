function[] = train_data_causal(image_name,pad,h,w)
    [bwImg,nb,nw] = pixel_extract(image_name);
    
    figure();
    imshow(bwImg);
    title('MATLAB converted image');
    
    strb = sprintf('Original Image Black Pixels: %d', nb); 
    strw = sprintf('Original Image White Pixels: %d', nw);
    stres = sprintf('Image Resolution %d x %d', size(bwImg,1),size(bwImg,2));
    disp(stres);
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
    view(btree);
    
    
    %create random initial image
    
    %white
    %RImg = ones(size(pad_bwImg));
    
    %black
    %RImg = ones(size(pad_bwImg));
    
    %random
    RImg = randi([0,1],size(pad_bwImg));
    
    figure();
    imshow(RImg);
    title('Initial image');
    
    newX = zeros(1, h*(2*w+1)+w);
    %disp(rows);
    for iter = 1:20
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
                %inverse transform method
                a = rand(1);
                if a < score(1)
                    RImg(i,j) = 0;
                else
                    RImg(i,j) = 1;
                end
            end
        end
        figure();
        imshow(RImg);
        str = sprintf('Reconstructed image %d',iter);
        title(str);
    end
end