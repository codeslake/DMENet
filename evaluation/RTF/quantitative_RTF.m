function [mse_list, mse_out_avg, mae_list, mae_out_avg] = get_mse(method, noise_level)

    load convertion.mat

    gt_file_paths = sort(dir2(['gt/', noise_level]));
    out_file_paths = sort(dir2(['out/', method, '/',  noise_level]));

    mse_list = [];
    mse_out_avg = 0.;

    mae_list = [];
    mae_out_avg = 0.;

    psnr_list = [];
    psnr_avg = 0.;

    ssim_list = [];
    ssim_avg = 0.;

    file_num = length(gt_file_paths);
    for i = 1:file_num
        gt_file_path = char(gt_file_paths(i));
        out_file_path = char(out_file_paths(i));

        % read gt
        load(gt_file_path);
        gt = blurMap;
        
        if strcmp(method, 'RTF') == false
            gauss_psf = zeros(size(gt));
            for k=1:size(gt,1)
                for j=1:size(gt,2)
                    [~, inda] = min(abs(convertion(:,2) - gt(k,j)));
                    gauss_psf(k,j) = convertion(inda,1);
                end
            end
            gt = gauss_psf;
        end

        % read outputs of mehtods
        if strcmp(method, 'RTF') == false
            out = double(imread(out_file_path));
            out = out/255.;
        end
        if strcmp(method, 'BDCS')
             % compute simga
            out = ((out * 15) - 1)/2;
            out(out < 0) = 0;
            % clip
            out(out > 3.275) = 3.275;
            out = out / 3.275;
            gt = gt / 3.275;
        end 
        
        out_shape = size(out);
        gt_shape = size(gt);
        new_shape = [min(out_shape(1), gt_shape(1)), min(out_shape(2),gt_shape(2))];

        gt = gt(1:new_shape(1), 1:new_shape(2));
        out = out(1:new_shape(1), 1:new_shape(2));
     
        % compute MSE
        % MSE_out = immse(gt, out);
        MSE_out = ((gt - out).^2);
        MSE_out = mean(MSE_out(:));
        mse_list = [mse_list; MSE_out];
        mse_out_avg = mse_out_avg + MSE_out;

        % compute MAE
        MAE_out = abs(gt - out);
        MAE_out = mean(MAE_out(:));
        mae_list = [mae_list; MAE_out];
        mae_out_avg = mae_out_avg + MAE_out;

    end
    mse_out_avg = mse_out_avg / file_num;
    mae_out_avg = mae_out_avg / file_num;

    %disp(['[', method, '] ', 'MAE: ', num2str(mae_out_avg), ' MSE: ', num2str(mse_out_avg)]);
end
