
clear; clc; close all;

%% Global Parameters
rng(42);
N_fft = 64;
N_cp = 16;
M = 4;                  % BS antennas
N_x = 16; N_y = 16;     % RIS grid
N = N_x * N_y;          % 256 RIS elements
L = 20;                 % Number of paths
mod_order = 4;
k_bits = log2(mod_order);
max_delay = 16;

num_samples = 250000;
num_train = 200000;
num_val = 50000;




%% Pre-allocating and Vectorize RIS Grid positions
X_data = zeros(num_samples, 256);
Y_data = zeros(num_samples, 128);

% Vectorizing array element positions for blazing fast channel generation
alpha_grid = repmat((0:N_x-1)', N_y, 1);
beta_grid = repelem((0:N_y-1)', N_x, 1);
m_grid = (0:M-1)';

% Single broadcasted pilot symbol
saved_pilots = qammod(randi([0 1], N_fft*k_bits, 1), mod_order, 'InputType','bit','UnitAveragePower',true);
tx_p_base = ifft(saved_pilots, N_fft) * sqrt(N_fft);
tx_p = tx_p_base / sqrt(M); % Power distributed across M antennas

%% Generation Training Data
for i = 1:num_samples
    % Fast Cascaded Channel Generation
    omega = rand(N, 1) * 2 * pi;
    Phi = diag(exp(1i * omega));
    
    alpha_1 = (randn(1, L) + 1i*randn(1, L)) / sqrt(2);
    tau_1 = randi([0, 7], 1, L); 
    phi_r_1 = rand(1, L) * pi; theta_r_1 = rand(1, L) * pi; 
    phi_t_1 = rand(1, L) * pi; 
    
    alpha_2 = (randn(1, L) + 1i*randn(1, L)) / sqrt(2);
    tau_2 = randi([0, 8], 1, L); 
    phi_t_2 = rand(1, L) * pi; theta_t_2 = rand(1, L) * pi; 
    
    % Vectorized Array Responses
    a_BS = exp(1i * pi * m_grid * sin(phi_t_1)) / sqrt(M);
    a_RIS_H1 = exp(1i * pi * (alpha_grid * (sin(phi_r_1).*sin(theta_r_1)) + beta_grid * cos(theta_r_1))) / sqrt(N);
    a_RIS_H2 = exp(1i * pi * (alpha_grid * (sin(phi_t_2).*sin(theta_t_2)) + beta_grid * cos(theta_t_2))) / sqrt(N);
    
    % Building Frequency Domain Cascaded Channel directly
    Heff_freq = zeros(1, N_fft);
    for k_sub = 0:N_fft-1
        H1_k = (a_RIS_H1 .* alpha_1 .* exp(-1i * 2 * pi * k_sub * tau_1 / N_fft)) * a_BS';
        H2_k = sum(a_RIS_H2 .* alpha_2 .* exp(-1i * 2 * pi * k_sub * tau_2 / N_fft), 2).';
        
        H1_k_scaled = sqrt(M*N/L) * H1_k;
        H2_k_scaled = sqrt(N/L) * H2_k;
        
        % Composite effective channel for broadcasted signal
        Heff_freq(k_sub+1) = H2_k_scaled * Phi * sum(H1_k_scaled, 2);
    end
    
    Heff_time = ifft(Heff_freq, N_fft);
    Heff_time = Heff_time(1:N_cp); % Truncating
    
    % Normalizing Channel Power
    chan_power = sum(abs(Heff_time).^2);
    Heff_time = Heff_time / sqrt(chan_power);
    
    % Generating Data Symbol
    data_bits = randi([0 1], N_fft*k_bits, 1);
    data_sym  = qammod(data_bits, mod_order, 'InputType','bit','UnitAveragePower',true);
    tx_d = (ifft(data_sym, N_fft) * sqrt(N_fft)) / sqrt(M);
    
    % Building Frame & Transmit
    frame = [tx_p(end-N_cp+1:end); tx_p; 
             tx_d(end-N_cp+1:end); tx_d];
             
    snr_db = randi([0, 30]);
    noise_var = 10^(-snr_db/10);
    
    rx = filter(Heff_time, 1, frame) + sqrt(noise_var/2) * (randn(size(frame)) + 1i*randn(size(frame)));
    
    %Extracting & Storing
    start_p = N_cp + 1;
    start_d = (N_fft+N_cp) + N_cp + 1;
    
    r_p = fft(rx(start_p : start_p+N_fft-1)) / sqrt(N_fft);
    r_d = fft(rx(start_d : start_d+N_fft-1)) / sqrt(N_fft);
    
    X_data(i,:) = [real(r_p)', imag(r_p)', real(r_d)', imag(r_d)'];
    Y_data(i,:) = data_bits';
end


%% Training 8 Chunk Networks
net_array = cell(1,8);
info_chunk1 = [];
for chunk = 1:8
    fprintf('Training Chunk %d/8 (Advanced Architecture)\n', chunk);
    Y_chunk = Y_data(:, (chunk-1)*16+1 : chunk*16);
    
    
    layers = [
        featureInputLayer(256, 'Normalization','zscore', 'Name','input')
        
        fullyConnectedLayer(1024, 'Name','fc1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','r1')
        dropoutLayer(0.3, 'Name', 'drop1') 
        
        fullyConnectedLayer(512, 'Name','fc2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','r2')
        dropoutLayer(0.2, 'Name', 'drop2') % Kills 20% of neurons
        
        fullyConnectedLayer(256, 'Name','fc3')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','r3')
        
        fullyConnectedLayer(16,  'Name','out')
        sigmoidLayer('Name','sig')
        regressionLayer('Name','reg')
    ];
    

    options = trainingOptions('adam', ...
        'MaxEpochs',        40, ...
        'MiniBatchSize',    1024, ...
        'InitialLearnRate', 0.002, ...
        'LearnRateSchedule','piecewise', ...    
        'LearnRateDropPeriod', 15, ...          
        'LearnRateDropFactor', 0.5, ...        
        'L2Regularization', 1e-4, ...           
        'ValidationData',   {X_data(num_train+1:end,:), Y_chunk(num_train+1:end,:)}, ...
        'ValidationFrequency', 50, ...
        'Plots',    'none', ...
        'Verbose',  true); 
        
    [net_array{chunk}, t_info] = trainNetwork(X_data(1:num_train,:), Y_chunk(1:num_train,:), layers, options);
    
    if chunk == 1
        info_chunk1 = t_info;  
    end
end
save('model_partB_dnn.mat', 'net_array', 'saved_pilots');
fprintf('\n=== Task 3 Complete! Advanced Model Saved. ===\n');

%%  Validation Plot
val_rmse = info_chunk1.ValidationRMSE(~isnan(info_chunk1.ValidationRMSE));
val_mse  = val_rmse .^ 2;
val_epochs = linspace(1, 40, length(val_mse));

figure('Name', 'Part B Task 3: Validation Loss');
plot(val_epochs, val_mse, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
title('Part B Task 3: Validation Loss (L_2 Loss) vs. Epochs');
xlabel('Epoch'); ylabel('Loss (MSE)');
legend('Validation Loss', 'Location', 'northeast');
ylim([0 max(val_mse)*1.1]);

dim = [.3 .6 .3 .3];
str = {'Input Data Size: 250,000 Samples', 'Input Feature Size: 256', 'Target Size: 16 Bits (Per Chunk)'};
annotation('textbox',dim,'String',str,'FitBoxToText','on', 'BackgroundColor', 'w');