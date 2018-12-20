% Annotation is performed by recording the coordinates of mouse
% click (on head) for each image. Generates ground truth .mat files
% for each image file using head annotations. True count is number 
% of entries in .mat file.

clear all
clc
close all
filePattern = fullfile('images', '/*.jpg');
ImageFiles = dir(filePattern);
n = length(ImageFiles);
read_path = 'images/';
store_path = 'ground-truth/';
t = 0;                         %number of files initially in training set

for i=1:n
    disp(i);
    im = imread([read_path 'IMG_' num2str(t+i) '.jpg']); 
    figure
    imshow(im)
    [x,y] = getpts;
    image_info{1,1}.location = [x y];
    image_info{1,1}.number = size(x,1);
    save([store_path 'GT_IMG_' num2str(t+i) '.mat'], 'image_info')
    close
end
