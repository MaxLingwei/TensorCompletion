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

addpath('toolbox\');
addpath('tensor_toolbox\');

for test_img_index=8
    myName=sprintf('TestImages/%s.bmp',filename{test_img_index});
    A=imread(myName);
    %figure(1);  imshow(A);

    myrate=0.60:0.05:0.95;
    myResult=cell(2,numel(myrate)); % store the completion result;
    A=double(A)/255.0;
    
    for iterate=1:numel(myrate)
        rate=1 - myrate(iterate);
        [row, col, channel]=size(A);
        B=zeros([row, col, channel]);
        mark=true([row, col, channel]);
        rand_set = rand(size(A));
        A_tensor = tensor(A);
        
        % omit some pixs by missing rate
        index = find(rand_set < rate);
        [a, b, c] = ind2sub([3, row, col], index);
        index = [c, b, a];
        value = A_tensor(index);
        value = value';

        % test code             
        MODE_2 = TVRTC_I(index, value, A);

        MODE_2_update = TVRTC_II(index, value, A);

        myResult{1,iterate}=MODE_2;  
        myResult{2,iterate}=MODE_2_update; 

        myName=['Result/' filename{test_img_index} '_Result.mat'];
        save(myName,'myResult','myrate');

        myName=['Result/' filename{test_img_index}, '_Data.mat'];
        save(myName,'A');
    end
end

[perform_RSE,perform_PSNR] = performance_eval();
