close all;
clear all;

filename=cell(8,1);
filename{1}='airplane';
filename{2}='baboon';
filename{3}='barbara';
filename{4}='facade';
filename{5}='house';
filename{6}='lena';
filename{7}='peppers';
filename{8}='sailboat';
filename{9}='pubu';
filename{10}='sun';
filename{11}='cloud';
filename{12}='balloons_RGB';

addpath('toolbox\');
addpath('tensor_toolbox\');
myrate=0.60:0.05:0.95;
result = cell(8, 8);
result_psnr = cell(8, 8);
result_rse = cell(8, 8);
for img_idx = 1:8
    myName=sprintf('TestImages/%s.bmp',filename{img_idx});
    A=imread(myName);
    A=double(A)/255.0;
    [row, col, channel] = size(A);
    for iter = 1:numel(myrate)
        sample_ratio = 1 - myrate(iter);
        
        rand_set = rand(size(A));
        A_tensor = tensor(A);
        
        
        index = find(rand_set < sample_ratio);
        [a, b, c] = ind2sub([3, 256, 256], index);
        index = [c, b, a];
        value = A_tensor(index);
        value = value';
    
        result{iter, img_idx} = TVRTC_II(index, value, A);
        result_rse{iter, img_idx} = rse(A, result{iter, img_idx});
        result_psnr{iter, img_idx} = psnr(A, result{iter, img_idx});
    end
    
end