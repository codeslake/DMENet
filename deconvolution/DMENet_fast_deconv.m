function [deconved, est_time] = fast_deconvolution(image, defocus_map, lambda, is_gpu)
    % Test the fast deconvolution  method presented in the paper
    % D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
    % Priors", Proceedings of NIPS 2009.

    %% parameter values; other values such as the continuation regime of the parameter beta should be changed in fast_deconv.m
    alpha = 1.5;
  

    %% my params
    g_size = 101;
    
    %% deconv start            

    unique_sigma = unique(defocus_map);

    g_center = (g_size + 1) / 2.;
    
    output = zeros(size(image));
    if is_gpu
        output = gpuArray(output);
    end
    tic;
    parfor (c = 1:3)
%     for c = 1:3

        image_temp = image(:, :, c);
        output_temp = ones(size(image_temp));
        if is_gpu
            output_temp = gpuArray(output_temp);
        end
        
        for j = 1:length(unique_sigma)
%             disp(sprintf('I[%02d/%02d], C[%d/%d], U[%03d/%03d]', i, length(image_file_paths), c, 3, j, length(unique_sigma)));
            s = unique_sigma(j);
            sigma = s;
            if s ~= 0
                G = fspecial('gaussian',[g_size, g_size], sigma);
            else
                G = zeros(g_size, g_size);
                G(g_center, g_center) = 1.0;
            end
            is_identity = G(g_center - 1:g_center+1, g_center - 1:g_center+1);
            is_identity(2, 2) = 0;
            is_identity = sum(is_identity(:)) == 0;

            if is_identity == false
                G_idx = find(G > 0);
                [y_G,x_G] = ind2sub(size(G),G_idx);
                kernel = G(min(y_G):max(y_G), min(x_G):max(x_G));
                ks = floor((size(kernel, 1) - 1)/2);

                image_pad = padarray(image_temp, [1 1]*ks, 'replicate', 'both');

                logical_G = logical(G > 0);
                if sum(logical_G(:)) > 1
                    for a=1:4
                      image_pad = edgetaper(image_pad, kernel);
                    end
                end
                if is_gpu
                    image_pad = gpuArray(image_pad);
                end

                % fast_deconv
                [x] = fast_deconv(image_pad, kernel, lambda, alpha);

                x = x(ks+1:end-ks, ks+1:end-ks);
            else
                x = image_temp;
            end

            output_temp(logical(defocus_map == s)) = x(logical(defocus_map == s));
        end
        output(:, :, c) = output_temp;
    end
    est_time = toc();

    output(logical(output > 1)) = 1;
    output(logical(output < 0)) = 0;
    if is_gpu
        output = gather(output);
    end
    deconved = output;

