clear all;
close all;
datasets = ["BDCS"];
noises = ["0", "1", "1.6"];

mse_list_total = [];
mse_avg_total = [];
mae_list_total = [];
mae_avg_total = [];
psnr_list_total = [];
psnr_avg_total = [];
ssim_list_total = [];
ssim_avg_total = [];
for i = 1:length(noises)
    mse_avg_temp = [];
    mae_avg_temp = [];
    psnr_avg_temp = [];
    ssim_avg_temp = [];
    for j = 1:length(datasets)
        [mse_list, mse_avg, mae_list, mae_avg] = quantitative_RTF(char(datasets(j)), char(noises(i)));
        
        if strcmp(noises(i), '0')
            mse_list_total = [mse_list_total, mse_list];
            mae_list_total = [mae_list_total, mae_list];
        end
        mse_avg_temp = [mse_avg_temp, mse_avg];
        mae_avg_temp = [mae_avg_temp, mae_avg];
    end
    mse_avg_total = [mse_avg_total; mse_avg_temp];
    mae_avg_total = [mae_avg_total; mae_avg_temp];
end

mae_list_total

disp(datasets)
disp("MSE (top to bottom: 0, 1, 1.6)")
disp(mse_avg_total)
disp("MAE (top to bottom: 0, 1, 1.6)")
disp(mae_avg_total)

