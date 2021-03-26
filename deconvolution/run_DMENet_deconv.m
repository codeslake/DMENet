function run_DMENet_deconv(dataset, is_gpu, gpu_num)
    % dataset =  CUHK | DPDD | RealDOF
    if nargin == 1
        is_gpu = false;
        gpu_num = 0;
    elseif nargin == 2
        gpu_num = 1;
    end

    g = 0;    
    if is_gpu
        g = gpuDevice(gpu_num);
    end

    %% directory
    offset = './sources';
    image_file_paths = dir2(fullfile(offset, 'input', dataset));
    defocus_file_paths = dir2(fullfile(offset, 'defocus_map', dataset));
    out_offset = fullfile('output', dataset);

    % remove output directory before begin
    if isdir(out_offset)
        rmdir(out_offset, 's');
    end
    mkdir(out_offset)

    %% deconv parameter
    if contains(dataset, 'CUHK')
        % for the CUHK dataset
        lambda = 420;
    elseif contains(dataset, 'DPDD')
        % for the DPDD dataset
        lambda = 16.8;
    elseif contains(dataset, 'RealDOF')
        % for the RealDOF dataset
        lambda = 4.2;
    end

    % my parameter
    bin_num = 32; % bin_number

    %% deconv start
    est_time_mean = 0;
    for i = 1:length(image_file_paths)
        % read images
        input = read_img(image_file_paths(i));
        
        %%%% Read defocus map and make it to sigma map
        %%% This is for defocus map result of DMENet. For the results of other
        %%% methods, modify them to have proper standard deviation value for
        %%% creating spatially varying Gaussian kernels.
        defocus_map = double(imread(char(defocus_file_paths(i))))./255.0;
        % DMENET
        defocus_map = (defocus_map * 15 - 1)/2;
        defocus_map(defocus_map < 0) = 0;
        % EBDB
        % defocus_map = defocus_map * 5;
        % JNB
        % defocus_map = defocus_map * 4;
        %%%%
        
        % make sure to have the same resolution
        [input, defocus_map] = refine_img(input, defocus_map);
        
        %%% quantize (for all compared methods, results are almost the same even without the qunatization when bin_num is more than 32)
        unique_sigma = unique(defocus_map);
        w_edges = quantile(unique_sigma, bin_num); % weighted edges
        w_edges = [min(unique_sigma), w_edges];
        w_edges = [w_edges, max(unique_sigma)];
        defocus_map_index = discretize(defocus_map, w_edges);
        defocus_map = w_edges(defocus_map_index);
        unique_sigma = unique(defocus_map);
        
        
        %%% deconvolution start
        [deconv_result, est_time] = DMENet_fast_deconv(input, defocus_map, lambda, is_gpu);
        if is_gpu
            reset(g);
        end
        %%%
        
        disp(sprintf('[%02d/%02d] (%.3f sec)', i, length(image_file_paths), est_time));
        est_time_mean = est_time_mean + est_time;
        
        imwrite(uint8(deconv_result*255), fullfile(out_offset, sprintf('%02d.png', i)));
    end

    est_time_mean = est_time_mean / length(image_file_paths);
    disp(sprintf('Deconvolution done for %s dataset (%.3f sec)', dataset, est_time_mean));
end

%%    
function image = read_img(path)
    image = imread(char(path));
    image = im2double(image);
    image = double(uint8(image * 255)) / 255;
end

function [in1, in2] = refine_img(in1, in2)
    sz_in1 = size(in1);
    sz_in2 = size(in2);
    
    in1 = in1(1:min(sz_in1(1), sz_in2(1)), 1:min(sz_in1(2), sz_in2(2)), :);
    in2 = in2(1:min(sz_in1(1), sz_in2(1)), 1:min(sz_in1(2), sz_in2(2)), :);
end
