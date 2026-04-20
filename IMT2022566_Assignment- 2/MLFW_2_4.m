
clear; clc; close all;

%% Loading DNN Models from Task 3
fprintf('Loading Part B DNN models\n');
try
    load('model_partB_dnn.mat'); 
    fprintf('DNN Models loaded successfully.\n\n');
catch
    error('model_partB_dnn.mat not found! Run Task 3 first.');
end

%% System Parameters
rng(42); 
N_fft = 64;             
N_cp = 16;              
M = 4;                  % BS antennas
K = 1;                  % User antenna
N_x = 16; N_y = 16;     % RIS grid
N = N_x * N_y;          % 256 elements
L = 20;                 % Multipath delays
max_delay = 16;
mod_order = 4;          % QPSK
k_bits = log2(mod_order);
SNR_vec = 0:5:30;
num_snr = length(SNR_vec);
num_iter = 1000;        % Iterations per SNR

%% Pre-computing Matrices & Allocations
tau = 0:max_delay-1;
pdp = exp(-tau/2); pdp = pdp / sum(pdp);
R_HH = zeros(N_fft, N_fft);
for r = 1:N_fft
    for c = 1:N_fft
        for l = 1:max_delay
            R_HH(r,c) = R_HH(r,c) + pdp(l)*exp(-1i*2*pi*(r-c)*(l-1)/N_fft);
        end
    end
end
W = hadamard(M); % Walsh-Hadamard for orthogonal pilots

ser_ls = zeros(num_snr, 1);
ser_mmse = zeros(num_snr, 1);
ser_dnn = zeros(num_snr, 1);

% Vectorized RIS Grid
alpha_grid = repmat((0:N_x-1)', N_y, 1);
beta_grid = repelem((0:N_y-1)', N_x, 1);
m_grid = (0:M-1)';

% DNN Pilot Symbol 
tx_p_dnn = (ifft(saved_pilots, N_fft) * sqrt(N_fft)) / sqrt(M);

%% Main Monte Carlo Loop
for si = 1:num_snr
    snr_db = SNR_vec(si);
    noise_var = 10^(-snr_db/10);
    noise_std = sqrt(noise_var/2);
    
    W_MMSE = R_HH / (R_HH + (noise_var/M) * eye(N_fft));
    
    err_ls = 0; err_mmse = 0;
    
    X_dnn_batch = zeros(num_iter, 256);
    Y_true_bits = zeros(num_iter, 128);
    
    for it = 1:num_iter
        % Vectorized mmWave Channel Generation
        omega = rand(N, 1) * 2 * pi;
        Phi = diag(exp(1i * omega));
        
        alpha_1 = (randn(1, L) + 1i*randn(1, L)) / sqrt(2);
        tau_1 = randi([0, 7], 1, L); phi_r_1 = rand(1, L) * pi; theta_r_1 = rand(1, L) * pi; phi_t_1 = rand(1, L) * pi; 
        
        alpha_2 = (randn(1, L) + 1i*randn(1, L)) / sqrt(2);
        tau_2 = randi([0, 8], 1, L); phi_t_2 = rand(1, L) * pi; theta_t_2 = rand(1, L) * pi; 
        
        a_BS = exp(1i * pi * m_grid * sin(phi_t_1)) / sqrt(M);
        a_RIS_H1 = exp(1i * pi * (alpha_grid * (sin(phi_r_1).*sin(theta_r_1)) + beta_grid * cos(theta_r_1))) / sqrt(N);
        a_RIS_H2 = exp(1i * pi * (alpha_grid * (sin(phi_t_2).*sin(theta_t_2)) + beta_grid * cos(theta_t_2))) / sqrt(N);
        
        Heff_freq = zeros(1, N_fft);
        for k_sub = 0:N_fft-1
            H1_k = (a_RIS_H1 .* alpha_1 .* exp(-1i * 2 * pi * k_sub * tau_1 / N_fft)) * a_BS';
            H2_k = sum(a_RIS_H2 .* alpha_2 .* exp(-1i * 2 * pi * k_sub * tau_2 / N_fft), 2).';
            Heff_freq(k_sub+1) = (sqrt(N/L) * H2_k) * Phi * sum(sqrt(M*N/L) * H1_k, 2);
        end
        
        Heff_time = ifft(Heff_freq, N_fft);
        Heff_time = Heff_time(1:N_cp); 
        
        % Normalized Channel
        chan_power = sum(abs(Heff_time).^2);
        Heff_time = Heff_time / sqrt(chan_power);
        
        % Shared Data Generation
        data_bits = randi([0 1], 128, 1);
        data_sym = qammod(data_bits, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);
        tx_d_broad = (ifft(data_sym, N_fft) * sqrt(N_fft)) / sqrt(M);
        Y_true_bits(it, :) = data_bits';
        
        % LS and MMSE (4 Orthogonal Pilots)
        num_pilot_sym = M;
        pilot_sym_ls = qammod(randi([0 mod_order-1], N_fft, 1), mod_order, 'UnitAveragePower', true);
        
        tx_grid_pilot = zeros(M, N_fft, num_pilot_sym);
        for sym_idx = 1:num_pilot_sym
            for tx_ant = 1:M
                tx_grid_pilot(tx_ant, :, sym_idx) = W(tx_ant, sym_idx) * pilot_sym_ls.';
            end
        end
        
        tx_signal_ls = zeros(M, (num_pilot_sym + 1) * (N_fft + N_cp));
        for sym_idx = 1:num_pilot_sym+1
            for tx_ant = 1:M
                if sym_idx <= num_pilot_sym
                    sym_freq = squeeze(tx_grid_pilot(tx_ant, :, sym_idx)).';
                    sym_time = ifft(sym_freq, N_fft) * sqrt(N_fft);
                else
                    sym_time = tx_d_broad; % 1 Data symbol
                end
                sym_time_cp = [sym_time(end-N_cp+1:end); sym_time];
                
                start_idx = (sym_idx-1)*(N_fft + N_cp) + 1;
                end_idx = sym_idx*(N_fft + N_cp);
                tx_signal_ls(tx_ant, start_idx:end_idx) = sym_time_cp.';
            end
        end
        
        % MISO Filter
        rx_signal_clean = zeros(1, size(tx_signal_ls, 2));
        for tx_ant = 1:M
            rx_signal_clean = rx_signal_clean + filter(Heff_time, 1, tx_signal_ls(tx_ant, :));
        end
        rx_ls = rx_signal_clean + noise_std * (randn(size(rx_signal_clean)) + 1i*randn(size(rx_signal_clean)));
        
        % LS/MMSE Receiver
        rx_grid = zeros(N_fft, num_pilot_sym + 1);
        for sym_idx = 1:num_pilot_sym+1
            start_idx = (sym_idx-1)*(N_fft + N_cp) + N_cp + 1;
            end_idx = sym_idx*(N_fft + N_cp);
            rx_grid(:, sym_idx) = fft(rx_ls(start_idx:end_idx).', N_fft) / sqrt(N_fft);
        end
        
        rx_pilots = rx_grid(:, 1:num_pilot_sym);
        H_LS = zeros(M, N_fft);
        for tx_ant = 1:M
            dec_pilot = zeros(N_fft, 1);
            for sym_idx = 1:num_pilot_sym
                dec_pilot = dec_pilot + W(tx_ant, sym_idx) * rx_pilots(:, sym_idx);
            end
            H_LS(tx_ant, :) = ((dec_pilot / num_pilot_sym) ./ pilot_sym_ls).';
        end
        
        H_MMSE = zeros(M, N_fft);
        for tx_ant = 1:M
            H_MMSE(tx_ant, :) = (W_MMSE * H_LS(tx_ant, :).').';
        end
        
        H_LS_comb = sum(H_LS, 1) / sqrt(M);
        H_MMSE_comb = sum(H_MMSE, 1) / sqrt(M);
        
        Y_d = rx_grid(:, end).'; 
        X_est_LS = Y_d ./ H_LS_comb;
        X_est_MMSE = Y_d ./ H_MMSE_comb;
        
        % Demodulate to bits
        rx_bits_ls = qamdemod(X_est_LS.', mod_order, 'OutputType', 'bit', 'UnitAveragePower', true);
        rx_bits_mmse = qamdemod(X_est_MMSE.', mod_order, 'OutputType', 'bit', 'UnitAveragePower', true);
        
        
        % Convert 128 bits to 64 QPSK symbols
        sym_true = data_bits(1:2:end)*2 + data_bits(2:2:end);
        sym_ls   = rx_bits_ls(1:2:end)*2 + rx_bits_ls(2:2:end);
        sym_mmse = rx_bits_mmse(1:2:end)*2 + rx_bits_mmse(2:2:end);

        % Calculate Symbol Errors
        err_ls = err_ls + sum(sym_ls ~= sym_true);
        err_mmse = err_mmse + sum(sym_mmse ~= sym_true);
        
        % DNN (1 Broadcasted Pilot)
        frame_dnn = [tx_p_dnn(end-N_cp+1:end); tx_p_dnn; 
                     tx_d_broad(end-N_cp+1:end); tx_d_broad];
                     
        rx_dnn = filter(Heff_time, 1, frame_dnn) + noise_std * (randn(size(frame_dnn)) + 1i*randn(size(frame_dnn)));
        
        start_p = N_cp + 1;
        start_d = (N_fft+N_cp) + N_cp + 1;
        
        r_p = fft(rx_dnn(start_p : start_p+N_fft-1)) / sqrt(N_fft);
        r_d = fft(rx_dnn(start_d : start_d+N_fft-1)) / sqrt(N_fft);
        
        X_dnn_batch(it, :) = [real(r_p)', imag(r_p)', real(r_d)', imag(r_d)'];
    end
    
    % Evaluating Chunked DNN Models
    pred_bits = zeros(num_iter, 128);
    for chunk = 1:8
        pred_bits(:, (chunk-1)*16+1:chunk*16) = double(predict(net_array{chunk}, X_dnn_batch) > 0.5);
    end
    

    err_dnn = 0;
    for it = 1:num_iter
        true_bits_dnn = Y_true_bits(it, :)';
        pred_bits_dnn = pred_bits(it, :)';
        
        % Convert bits to symbols for DNN
        sym_true_dnn = true_bits_dnn(1:2:end)*2 + true_bits_dnn(2:2:end);
        sym_pred_dnn = pred_bits_dnn(1:2:end)*2 + pred_bits_dnn(2:2:end);
        
        err_dnn = err_dnn + sum(sym_pred_dnn ~= sym_true_dnn);
    end
    
    % Calculating Final SER (64 symbols instead of 128 bits)
    total_symbols = num_iter * 64;
    ser_ls(si) = err_ls / total_symbols;
    ser_mmse(si) = err_mmse / total_symbols;
    ser_dnn(si) = err_dnn / total_symbols;
    
    fprintf('SNR = %2ddB | LS: %.4f | MMSE: %.4f | DNN: %.4f\n', snr_db, ser_ls(si), ser_mmse(si), ser_dnn(si));
end

%% Plotting Final Results
figure('Name', 'Part B Task 4: Final SER vs SNR');
semilogy(SNR_vec, ser_ls, 'r-o', 'LineWidth', 2, 'DisplayName', 'LS Estimator'); hold on;
semilogy(SNR_vec, ser_mmse, 'b-s', 'LineWidth', 2, 'DisplayName', 'MMSE Estimator');
semilogy(SNR_vec, ser_dnn, 'k-^', 'LineWidth', 2, 'DisplayName', 'Deep Learning (DNN)');
grid on; xlabel('SNR (dB)'); ylabel('Symbol Error Rate (SER)');
title('RIS-Assisted MU-MISO: SER vs SNR Comparison');
legend('Location', 'southwest'); ylim([1e-4 1]);