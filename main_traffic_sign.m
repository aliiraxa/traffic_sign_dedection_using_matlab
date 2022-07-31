clear all
close all

% first of all run training file than come to main file
% Reading input video
[filename, pathname] = uigetfile('*.*', 'select test video');
filewithpath=strcat(pathname, filename);
load rcnn;

v = VideoReader(filewithpath);  %Declare video object
videoplayer=vision.VideoPlayer();  %Initialize video player

runloop= true;   %conditions for the while loop

while runloop
    
img = readFrame(v); %Reading one frame
    data=img;
    
      me=data;
    %getting read color
    red=data(:,:,1);

    % convert picture into gray
    gray=rgb2gray(data);

    % subtract red to gray frame
    diff_im=imsubtract(red,gray);

    % use median filter
    diff_im=medfilt2(diff_im,[3,3]);

    % convert image into binary
    diff_im=im2bw(diff_im,0.10);

    % use blob statistics analysis on this image 
    diff_im=bwareaopen(diff_im,100);
    
    % filling shapes  
    diff_im=imfill(diff_im,'hole');
    bw=bwlabel(diff_im,8);
    
    % use boundbox to enbox red color object
     stats=regionprops(bw,'BoundingBox','Centroid');

     %showing image for comparisan of machine learning and ojbect selection
     image(data);


     hold on

    % identify no of red object in image

    for(object=1:length(stats))

    bb=stats(object).BoundingBox;   

    bc=stats(object).Centroid;

    %creating reactangle 
    rectangle('Position',bb,'EdgeColor','r','LineWidth',2);

    %Ploting reactangle on Image 
    plot(bc(1),bc(2),'-m+');

    %Ploting Y and X on Image 
    a=text(bc(1)+15,bc(2),strcat('X: ',num2str(round(bc(1))),' Y: ',num2str(round(bc(2)))));
    set(a,'FontName','Arial','FontWeight','bold','FontSize',12,'Color','yellow');

    
    if(length(stats)==1)

    % show one with one object
    
    a=text(50,60,strcat('One'));

    set(a,'FontName','Arial','FontWeight','bold','FontSize',12,'Color','red');

    end


    end

    %ending code for area selected
    
    %machine learning classifications
    %stop sign already training by training folder
    [bbox, score, ~] = detect(rcnn, img, 'MiniBatchSize', 32);  %Detecting Stop sign
    [score1, idx]=max(score);   %getting max. Score
    if(score1==1)
 
    bbox1 = bbox(idx, :);  %getting Bounding box corresponding to max. Score
    img = insertObjectAnnotation(img, 'rectangle', bbox1,...
        strcat('Stop Sign: Conf. Score:', num2str(score1)));  %Insertig anotation
    end
    
    step(videoplayer, img); %Displaying image as frame in the video player
    runloop = isOpen(videoplayer);   %checking video player is ON or OFF
end