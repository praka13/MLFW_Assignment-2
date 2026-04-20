
clear; clc; close all;

%% Global Parameters
rng(42);
N_fft       = 64;
N_cp        = 16;
M           = 4;
k           = log2(M);           % 2 bits/symbol (QPSK)
max_delay   = 16;
tau         = 0:max_delay-1;
pdp         = exp(-tau/2);       
pdp         = pdp / sum(pdp);
pilot_idx_8 = 1:8:64;



%% Training All 3 Scenarios
fprintf('64 Pilots, With CP\n');
[info_64] = train_chunked_models('model_64_cp.mat', 64, true, ...
                                  N_fft, N_cp, M, k, pdp, pilot_idx_8);

fprintf('\n8 Pilots, With CP\n');
train_chunked_models('model_8_cp.mat', 8, true, ...
                      N_fft, N_cp, M, k, pdp, pilot_idx_8);

fprintf('\n64 Pilots, No CP\n');
train_chunked_models('model_64_nocp.mat', 64, false, ...
                      N_fft, N_cp, M, k, pdp, pilot_idx_8);



%% Plot Validation Loss
val_rmse = info_64.ValidationRMSE(~isnan(info_64.ValidationRMSE));
val_mse  = val_rmse .^ 2;
val_epochs = linspace(1, 40, length(val_mse));

figure('Name', 'Task 3: Validation Loss');
plot(val_epochs, val_mse, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
title('Task 3: Validation Loss (L_2 Loss) vs. Epochs');
xlabel('Epoch'); ylabel('Loss (MSE)');
legend('Validation Loss', 'Location', 'northeast');
ylim([0 max(val_mse)*1.1]);


%% FUNCTION

function [info_chunk1] = train_chunked_models(filename, num_pilots, use_cp, ...
                                               N_fft, N_cp, M, k, pdp, pilot_idx_8)
    num_samples = 250000;
    num_train   = 200000;   
    num_val     = 50000;    
    X_data = zeros(num_samples, 256);
    Y_data = zeros(num_samples, 128);
    max_delay = length(pdp);
    
    %% Generating Fixed Pilots
    if num_pilots == 64
        saved_pilots = qammod(randi([0 1], N_fft*k, 1), M, ...
                              'InputType','bit','UnitAveragePower',true);
    else
        saved_pilots = qammod(randi([0 1], length(pilot_idx_8)*k, 1), M, ...
                              'InputType','bit','UnitAveragePower',true);
    end
    
    %% Generating Training Data
    
    for i = 1:num_samples
        % Random data bits and symbols
        data_bits = randi([0 1], N_fft*k, 1);
        data_sym  = qammod(data_bits, M, 'InputType','bit','UnitAveragePower',true);
        
        % Pilot block
        if num_pilots == 64
            tx_p = ifft(saved_pilots, N_fft) * sqrt(N_fft);
        else
            sym_8 = zeros(N_fft,1);
            sym_8(pilot_idx_8) = saved_pilots;
            sym_8(setdiff(1:N_fft,pilot_idx_8)) = qammod( ...
                randi([0 1],56*k,1), M,'InputType','bit','UnitAveragePower',true);
            tx_p = ifft(sym_8, N_fft) * sqrt(N_fft);
        end
        
        % Data block
        tx_d = ifft(data_sym, N_fft) * sqrt(N_fft);
        
        % Build frame (Pilot + Data ONLY)
        if use_cp
            frame = [tx_p(end-N_cp+1:end);     tx_p;       % pilot block + CP
                     tx_d(end-N_cp+1:end);     tx_d];      % data  block + CP
            start_p = N_cp + 1;                            % start of pilot
            start_d = (N_fft+N_cp) + N_cp + 1;             % start of data
        else
            frame   = [tx_p; tx_d];                        % NO CP - Real ISI occurs here
            start_p = 1;                                   % pilot starts immediately
            start_d = N_fft + 1;                           % data starts right after pilot
        end
        
        % Random channel
        h = sqrt(pdp/2).' .* (randn(max_delay,1) + 1j*randn(max_delay,1));
        
        % SNR range 0-30 dB
        snr_db   = randi([0, 30]);
        noise_var = 10^(-snr_db/10);
        
        rx = filter(h, 1, frame) + ...
             sqrt(noise_var/2) * (randn(size(frame)) + 1j*randn(size(frame)));
             
        % Extracting pilot and data blocks (DFT)
        r_p = fft(rx(start_p : start_p+N_fft-1)) / sqrt(N_fft);
        r_d = fft(rx(start_d : start_d+N_fft-1)) / sqrt(N_fft);
        
       
        X_data(i,:) = [real(r_p)', imag(r_p)', real(r_d)', imag(r_d)'];
        Y_data(i,:) = data_bits';
    end
    
    %% Training 8 Chunk Networks
    net_array  = cell(1,8);
    info_chunk1 = [];
    for chunk = 1:8
        fprintf('  -> Training Chunk %d/8...\n', chunk);
        Y_chunk = Y_data(:, (chunk-1)*16+1 : chunk*16);  
        
     
        layers = [
            featureInputLayer(256, 'Normalization','zscore', 'Name','input')
            fullyConnectedLayer(500, 'Name','fc1')
            reluLayer('Name','r1')
            fullyConnectedLayer(250, 'Name','fc2')
            reluLayer('Name','r2')
            fullyConnectedLayer(120, 'Name','fc3')
            reluLayer('Name','r3')
            fullyConnectedLayer(16,  'Name','out')
            sigmoidLayer('Name','sig')
            regressionLayer('Name','reg')
        ];
        
        options = trainingOptions('adam', ...
            'MaxEpochs',        40, ...
            'MiniBatchSize',    1024, ...
            'InitialLearnRate', 0.001, ...
            'L2Regularization', 1e-5, ...
            'ValidationData',   {X_data(num_train+1:end,:), ...
                                  Y_chunk(num_train+1:end,:)}, ...
            'ValidationFrequency', 50, ...
            'Plots',    'none', ...
            'Verbose',  false);
            
        [net_array{chunk}, t_info] = trainNetwork( ...
            X_data(1:num_train,:), Y_chunk(1:num_train,:), layers, options);
            
        if chunk == 1
            info_chunk1 = t_info;  
        end
    end
    
    save(filename, 'net_array', 'saved_pilots');
    fprintf('  Saved: %s\n', filename);
end