close all
clear all
clc

addpath('MRF');
%MRFParams = single([105 200 1.0]);% Shanghaitech Part_A
MRFParams = single([200 200 8]);% Shanghaitech Part_B
part = 'B';
load(['data/predictions_' part '_SHT.mat']);
load(['data/test_' part '_SHT.mat']);
load(['data/ground_truth_' part '_SHT.mat']);
read_path = ['ShanghaiTech/part_' part '/test_data/images/'];
store_path = 'result\';

n = numel(counts);
k = 1;
finalcount = zeros(n, 1);
for i = 1 : n
    disp(i);
    patchCount = counts{i};

    [height, width] = size(patchCount);
    p = reshape(predictions(k: k + height * width - 1), width, height);
    k = k + height * width;
    
    % The marginal data of the predicted count matrix is 0 after apply MRF, 
    % so first extending the predicted count matrix by copy marginal data.
    p = uint8(p)';
    p = [p(1,:); p];
    p = [p ;p(end,:)];
    p = [p(:, 1) p];
    p = [p p(:, end)];
    % apply MRF
    p = MRF(p, MRFParams);
    p = p(2:end-1, 2: end-1);
    
    finalcount(i) = FinalCount(p);
    
    im = imread([read_path 'IMG_' num2str(i) '.jpg']);
    figure
    imshow(im)
    title(['True count:' num2str(gt(i)) '   Predicted count:' num2str(finalcount(i))])
    saveas(gcf,[store_path 'IMG_' num2str(i) '.png']);
    close
end

MAE = mean(abs(finalcount - gt));
MSE = mean((finalcount - gt).^2)^0.5;
fprintf('MAE: %f\n', MAE);
fprintf('MSE: %f\n', MSE);
