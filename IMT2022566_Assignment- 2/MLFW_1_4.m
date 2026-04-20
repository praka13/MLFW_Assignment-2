clear; clc; close all;

%% Load Models

try
    m1 = load('model_64_cp.mat');   nets_64_cp   = m1.net_array; pilots_64_cp   = m1.saved_pilots;
    m2 = load('model_8_cp.mat');    nets_8_cp    = m2.net_array; pilots_8_cp    = m2.saved_pilots;
    m3 = load('model_64_nocp.mat'); nets_64_nocp = m3.net_array; pilots_64_nocp = m3.saved_pilots;
    fprintf('All 3 models loaded (8 chunks each).\n\n');
catch
    error('Models not found! Go and Run train_task3.m first.');
end

%% Parameters
rng(42);
N_fft     = 64;
N_cp      = 16;
M         = 4;
k         = log2(M);
num_bits  = N_fft * k;    
max_delay = 16;
SNR_vec   = 0:5:30;
num_snr   = length(SNR_vec);
num_iter  = 2000;
pilot_idx_8 = 1:8:64;
data_idx_8  = setdiff(1:N_fft, pilot_idx_8);

% PDP
tau = 0:max_delay-1;
pdp = exp(-tau/2);  
pdp = pdp/sum(pdp);

% Covariance Matrices
R_HH = zeros(N_fft,N_fft);
for r = 1:N_fft
    for c = 1:N_fft
        for l = 1:max_delay
            R_HH(r,c) = R_HH(r,c) + pdp(l)*exp(-1j*2*pi*(r-c)*(l-1)/N_fft);
        end
    end
end


R_Hp_Hp = R_HH(pilot_idx_8, pilot_idx_8);  
R_Hall_Hp = R_HH(:, pilot_idx_8);          
I_8 = eye(length(pilot_idx_8));

%% Pre-allocating BER Arrays
ber_ls_64    = zeros(num_snr,1); ber_mmse_64    = zeros(num_snr,1); ber_dnn_64    = zeros(num_snr,1);
ber_ls_8     = zeros(num_snr,1); ber_mmse_8     = zeros(num_snr,1); ber_dnn_8     = zeros(num_snr,1);
ber_ls_nocp  = zeros(num_snr,1); ber_mmse_nocp  = zeros(num_snr,1); ber_dnn_nocp  = zeros(num_snr,1);

%% Main Loop
fprintf('Starting BER evaluation (%d iterations per SNR)\n\n', num_iter);
for si = 1:num_snr
    snr_db    = SNR_vec(si);
    noise_var = 10^(-snr_db/10);
    noise_std = sqrt(noise_var/2);
    
    % Computing MMSE weights for this SNR
    W_MMSE_64 = R_HH / (R_HH + noise_var*eye(N_fft));
    W_MMSE_8  = R_Hall_Hp / (R_Hp_Hp + noise_var*I_8); 
    
    err_ls_64=0; err_mmse_64=0;
    err_ls_8=0;  err_mmse_8=0;
    err_ls_nocp=0; err_mmse_nocp=0;
    
    X_dnn_64   = zeros(num_iter,256);
    X_dnn_8    = zeros(num_iter,256);
    X_dnn_nocp = zeros(num_iter,256);
    Y_true     = zeros(num_iter,128);
    
    for it = 1:num_iter
        data_bits = randi([0 1], num_bits, 1);
        data_sym  = qammod(data_bits, M,'InputType','bit','UnitAveragePower',true);
        Y_true(it,:) = data_bits';
        
        tx_d = ifft(data_sym,  N_fft)*sqrt(N_fft);
        
        % 64 Pilots With CP
        tx_p64   = ifft(pilots_64_cp, N_fft)*sqrt(N_fft);
        frame_64 = [tx_p64(end-N_cp+1:end);   tx_p64;
                    tx_d(end-N_cp+1:end);      tx_d];
                    
        % 8 Pilots With CP
        sym_8 = zeros(N_fft,1);
        sym_8(pilot_idx_8) = pilots_8_cp;
        sym_8(data_idx_8)  = qammod(randi([0 1],length(data_idx_8)*k,1),...
                                    M,'InputType','bit','UnitAveragePower',true);
        tx_p8   = ifft(sym_8, N_fft)*sqrt(N_fft);
        frame_8 = [tx_p8(end-N_cp+1:end);    tx_p8;
                   tx_d(end-N_cp+1:end);      tx_d];
                   
        % 64 Pilots No CP
        tx_p_nocp  = ifft(pilots_64_nocp, N_fft)*sqrt(N_fft);
        frame_nocp = [tx_p_nocp; tx_d];
        
        % Channel
        h = sqrt(pdp/2).' .* (randn(max_delay,1)+1j*randn(max_delay,1));
        
        rx_64   = filter(h,1,frame_64)   + noise_std*(randn(size(frame_64))  +1j*randn(size(frame_64)));
        rx_8    = filter(h,1,frame_8)    + noise_std*(randn(size(frame_8))   +1j*randn(size(frame_8)));
        rx_nocp = filter(h,1,frame_nocp) + noise_std*(randn(size(frame_nocp))+1j*randn(size(frame_nocp)));
        
        start_p_cp = N_cp + 1;                             
        start_d_cp = (N_fft+N_cp) + N_cp + 1;              
        
        % 64 Pilots With CP
        r_p64 = fft(rx_64(start_p_cp : start_p_cp+N_fft-1)) / sqrt(N_fft);
        r_d64 = fft(rx_64(start_d_cp : start_d_cp+N_fft-1)) / sqrt(N_fft);
        
        H_LS_64   = r_p64 ./ pilots_64_cp;
        H_MMSE_64 = W_MMSE_64 * H_LS_64;
        
        err_ls_64   = err_ls_64   + sum(qamdemod(r_d64./H_LS_64,  M,'OutputType','bit','UnitAveragePower',true) ~= data_bits);
        err_mmse_64 = err_mmse_64 + sum(qamdemod(r_d64./H_MMSE_64,M,'OutputType','bit','UnitAveragePower',true) ~= data_bits);
        X_dnn_64(it,:) = [real(r_p64)', imag(r_p64)', real(r_d64)', imag(r_d64)'];
        
        % Process: 8 Pilots With CP
        r_p8 = fft(rx_8(start_p_cp : start_p_cp+N_fft-1)) / sqrt(N_fft);
        r_d8 = fft(rx_8(start_d_cp : start_d_cp+N_fft-1)) / sqrt(N_fft);
        
        % Direct MMSE and PCHIP Interpolation
        r_p8_raw = r_p8(pilot_idx_8);
        H_LS_8_raw = r_p8_raw ./ pilots_8_cp;
        
        H_LS_8  = interp1(pilot_idx_8, H_LS_8_raw, 1:N_fft,'pchip','extrap').';
        H_MMSE_8 = W_MMSE_8 * H_LS_8_raw; 
        
        err_ls_8   = err_ls_8   + sum(qamdemod(r_d8./H_LS_8,  M,'OutputType','bit','UnitAveragePower',true) ~= data_bits);
        err_mmse_8 = err_mmse_8 + sum(qamdemod(r_d8./H_MMSE_8,M,'OutputType','bit','UnitAveragePower',true) ~= data_bits);
        X_dnn_8(it,:) = [real(r_p8)', imag(r_p8)', real(r_d8)', imag(r_d8)'];
        
        % 64 Pilots No CP 
        start_p_nocp = 1;         
        start_d_nocp = N_fft + 1; 
        
        r_p_nocp = fft(rx_nocp(start_p_nocp : start_p_nocp+N_fft-1)) / sqrt(N_fft);
        r_d_nocp = fft(rx_nocp(start_d_nocp : start_d_nocp+N_fft-1)) / sqrt(N_fft);
        
        H_LS_nocp   = r_p_nocp ./ pilots_64_nocp;
        H_MMSE_nocp = W_MMSE_64 * H_LS_nocp;
        
        err_ls_nocp   = err_ls_nocp   + sum(qamdemod(r_d_nocp./H_LS_nocp,  M,'OutputType','bit','UnitAveragePower',true) ~= data_bits);
        err_mmse_nocp = err_mmse_nocp + sum(qamdemod(r_d_nocp./H_MMSE_nocp,M,'OutputType','bit','UnitAveragePower',true) ~= data_bits);
        X_dnn_nocp(it,:) = [real(r_p_nocp)', imag(r_p_nocp)', real(r_d_nocp)', imag(r_d_nocp)'];
    end 
    
    %% DNN Batch Prediction
    pred_64   = zeros(num_iter,128);
    pred_8    = zeros(num_iter,128);
    pred_nocp = zeros(num_iter,128);
    
    for chunk = 1:8
        bs = (chunk-1)*16+1;  be = chunk*16;
        pred_64(:,  bs:be) = double(predict(nets_64_cp{chunk},   X_dnn_64)   > 0.5);
        pred_8(:,   bs:be) = double(predict(nets_8_cp{chunk},    X_dnn_8)    > 0.5);
        pred_nocp(:,bs:be) = double(predict(nets_64_nocp{chunk}, X_dnn_nocp) > 0.5);
    end
    
    %% BER Calculation
    total = num_iter * num_bits;
    ber_ls_64(si)     = err_ls_64   / total;
    ber_mmse_64(si)   = err_mmse_64 / total;
    ber_dnn_64(si)    = sum(sum(pred_64   ~= Y_true)) / total;
    
    ber_ls_8(si)      = err_ls_8    / total;
    ber_mmse_8(si)    = err_mmse_8  / total;
    ber_dnn_8(si)     = sum(sum(pred_8    ~= Y_true)) / total;
    
    ber_ls_nocp(si)   = err_ls_nocp   / total;
    ber_mmse_nocp(si) = err_mmse_nocp / total;
    ber_dnn_nocp(si)  = sum(sum(pred_nocp ~= Y_true)) / total;
    
    fprintf('SNR=%2ddB | LS=%.4f  MMSE=%.4f  DNN=%.4f\n',...
            snr_db, ber_ls_64(si), ber_mmse_64(si), ber_dnn_64(si));
end

%% Plots
figure('Name','Task4 Fig1: 64 Pilots with CP');
semilogy(SNR_vec,ber_ls_64,  'r-o','LineWidth',1.5,'DisplayName','LS'); hold on;
semilogy(SNR_vec,ber_mmse_64,'b-s','LineWidth',1.5,'DisplayName','MMSE');
semilogy(SNR_vec,ber_dnn_64, 'k-^','LineWidth',2,  'DisplayName','Deep Learning (DNN)');
grid on; xlabel('SNR (dB)'); ylabel('BER');
title('Fig.1: BER vs SNR (64 Pilots, With CP)');
legend('Location','southwest'); ylim([1e-4 1]);

figure('Name','Task4 Fig2: 8 Pilots with CP');
semilogy(SNR_vec,ber_ls_8,  'm-o','LineWidth',1.5,'DisplayName','LS (Interpolated)'); hold on;
semilogy(SNR_vec,ber_mmse_8,'b-s','LineWidth',1.5,'DisplayName','MMSE (Projected)');
semilogy(SNR_vec,ber_dnn_8, 'k-^','LineWidth',2,  'DisplayName','Deep Learning (DNN)');
grid on; xlabel('SNR (dB)'); ylabel('BER');
title('Fig.2: BER vs SNR (8 Pilots, With CP)');
legend('Location','southwest'); ylim([1e-4 1]);

figure('Name','Task4 Fig3: Impact of CP Removal');
semilogy(SNR_vec,ber_ls_nocp,  'r--o','LineWidth',1.5,'DisplayName','LS Without CP'); hold on;
semilogy(SNR_vec,ber_mmse_nocp,'b--s','LineWidth',1.5,'DisplayName','MMSE Without CP');
semilogy(SNR_vec,ber_dnn_64,   'k-',  'LineWidth',2,  'DisplayName','DNN With CP (Reference)');
semilogy(SNR_vec,ber_dnn_nocp, 'k--^','LineWidth',2,  'DisplayName','DNN Without CP');
grid on; xlabel('SNR (dB)'); ylabel('BER');
title('Fig.3: Impact of CP Removal (64 Pilots)');
legend('Location','southwest'); ylim([1e-4 1]);
