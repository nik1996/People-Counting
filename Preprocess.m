# Perform data preprocessing by extracting features and performing annotation.

close all
clear all
clc

imgp1 = 'ShanghaiTech/part_B/train_data/images/';
gp1 = 'ShanghaiTech/part_B/train_data/ground-truth/';
label1 = 'train';

imgp2 = 'ShanghaiTech/part_B/test_data/images/';
gp2 = 'ShanghaiTech/part_B/test_data/ground-truth/';
label2 = 'test';

ExtractFeatures(imgp1, gp1, label1)
ExtractFeatures(imgp2, gp2, label2)

disp('Feature extraction is complete.');
