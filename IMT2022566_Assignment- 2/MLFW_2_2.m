
clear; clc; close all;

%% Parameters
rng(42); 
N_fft = 64;             
N_cp = 16;              
M = 4;                  % Number of BS antennas
K = 1;                  % Number of users (1 receive antenna)
N_x = 16; N_y = 16;     % RIS grid
N = N_x * N_y;          % 256 RIS elements
L = 20;                 % Multipath delays
mod_order = 4;          % QPSK
k_bits = log2(mod_order);

num_pilot_sym = M;      % 4 OFDM symbols for orthogonal pilots
num_data_sym = 10;      % Data symbols per frame
num_total_sym = num_pilot_sym + num_data_sym;

SNR_vec = 0:5:30;
num_snr = length(SNR_vec);
num_iter = 500;         % Iterations per SNR



%% Pre-compuing Channel Statistics for MMSE
max_delay = 16;
tau = 0:max_delay-1;
pdp = exp(-tau/2); 
pdp = pdp / sum(pdp);

R_HH = zeros(N_fft, N_fft);
for r = 1:N_fft
    for c = 1:N_fft
        for l = 1:max_delay
            R_HH(r,c) = R_HH(r,c) + pdp(l)*exp(-1i*2*pi*(r-c)*(l-1)/N_fft);
        end
    end
end

%% Pre-allocating SER Arrays
ser_ls = zeros(num_snr, 1);
ser_mmse = zeros(num_snr, 1);

% Applying Walsh-Hadamard Matrix for Orthogonal Pilots across M antennas
W = hadamard(M);

%% Main Monte Carlo Loop
for si = 1:num_snr
    snr_db = SNR_vec(si);
    noise_var = 10^(-snr_db/10);
    noise_std = sqrt(noise_var/2);
    
    % MMSE Filtering Matrix
    W_MMSE = R_HH / (R_HH + (noise_var/M) * eye(N_fft));
    
    err_ls = 0;
    err_mmse = 0;
    
    for it = 1:num_iter
        %Generate Cascaded Channel
        omega = rand(N, 1) * 2 * pi;
        Phi = diag(exp(1i * omega));
        
        alpha_1 = (randn(L, 1) + 1i*randn(L, 1)) / sqrt(2);
        tau_1 = randi([0, 7], L, 1); 
        phi_r_1 = rand(L, 1) * pi; theta_r_1 = rand(L, 1) * pi; 
        phi_t_1 = rand(L, 1) * pi; 
        
        alpha_2 = (randn(L, 1) + 1i*randn(L, 1)) / sqrt(2);
        tau_2 = randi([0, 8], L, 1); 
        phi_t_2 = rand(L, 1) * pi; theta_t_2 = rand(L, 1) * pi; 
        
        a_BS = @(phi) (1/sqrt(M)) * exp(1i * pi * (0:M-1)' * sin(phi));
        
        a_RIS_H1 = zeros(N, L); a_RIS_H2 = zeros(N, L); 
        for l = 1:L
            idx = 1;
            for beta = 0:N_y-1
                for alpha_idx = 0:N_x-1
                    a_RIS_H1(idx, l) = exp(1i * pi * (alpha_idx * sin(phi_r_1(l))*sin(theta_r_1(l)) + beta * cos(theta_r_1(l))));
                    a_RIS_H2(idx, l) = exp(1i * pi * (alpha_idx * sin(phi_t_2(l))*sin(theta_t_2(l)) + beta * cos(theta_t_2(l))));
                    idx = idx + 1;
                end
            end
        end
        a_RIS_H1 = (1/sqrt(N)) * a_RIS_H1; a_RIS_H2 = (1/sqrt(N)) * a_RIS_H2;
        
        H1_freq = zeros(N, M, N_fft); H2_freq = zeros(1, N, N_fft);
        for k_sub = 0:N_fft-1
            H1_k = zeros(N, M); H2_k = zeros(1, N);
            for l = 1:L
                H1_k = H1_k + alpha_1(l) * a_RIS_H1(:, l) * a_BS(phi_t_1(l))' * exp(-1i * 2 * pi * k_sub * tau_1(l) / N_fft);
                H2_k = H2_k + alpha_2(l) * a_RIS_H2(:, l)' * exp(-1i * 2 * pi * k_sub * tau_2(l) / N_fft);
            end
            H1_freq(:, :, k_sub+1) = sqrt(M*N/L) * H1_k;
            H2_freq(1, :, k_sub+1) = sqrt(N/L) * H2_k;
        end
        
        Heff_freq = zeros(1, M, N_fft);
        for k_sub = 1:N_fft
            Heff_freq(1, :, k_sub) = H2_freq(1, :, k_sub) * Phi * H1_freq(:, :, k_sub);
        end
        
        Heff_time = ifft(Heff_freq, N_fft, 3); 
        Heff_time = Heff_time(:, :, 1:N_cp);
        
        % Normalized channel power so SNR is mathematically accurate
        chan_power = sum(abs(Heff_time(:)).^2) / M;
        Heff_time = Heff_time / sqrt(chan_power);
        Heff_freq = Heff_freq / sqrt(chan_power);
        
        % Transmission Grid
        %Pilots
        pilot_sym = qammod(randi([0 mod_order-1], N_fft, 1), mod_order, 'UnitAveragePower', true);
        tx_grid_pilot = zeros(M, N_fft, num_pilot_sym);
        for sym_idx = 1:num_pilot_sym
            for tx_ant = 1:M
                tx_grid_pilot(tx_ant, :, sym_idx) = W(tx_ant, sym_idx) * pilot_sym.';
            end
        end
        
        %Data
        data_sym_raw = randi([0 mod_order-1], 1, N_fft, num_data_sym);
        data_sym = qammod(data_sym_raw, mod_order, 'UnitAveragePower', true);
        tx_grid_data = zeros(M, N_fft, num_data_sym);
        for tx_ant = 1:M
            tx_grid_data(tx_ant, :, :) = data_sym / sqrt(M); % Distribute power
        end
        
        tx_grid = cat(3, tx_grid_pilot, tx_grid_data);
        
        % IFFT & CP Addition
        tx_signal = zeros(M, num_total_sym * (N_fft + N_cp));
        for sym_idx = 1:num_total_sym
            for tx_ant = 1:M
                sym_freq = squeeze(tx_grid(tx_ant, :, sym_idx)).';
                sym_time = ifft(sym_freq, N_fft) * sqrt(N_fft);
                sym_time_cp = [sym_time(end-N_cp+1:end); sym_time];
                
                start_idx = (sym_idx-1)*(N_fft + N_cp) + 1;
                end_idx = sym_idx*(N_fft + N_cp);
                tx_signal(tx_ant, start_idx:end_idx) = sym_time_cp.';
            end
        end
        
        %Channel & Noise
        rx_signal_clean = zeros(1, size(tx_signal, 2));
        for tx_ant = 1:M
            h_tap = squeeze(Heff_time(1, tx_ant, :)); 
            rx_signal_clean = rx_signal_clean + filter(h_tap, 1, tx_signal(tx_ant, :));
        end
        
        rx_signal = rx_signal_clean + noise_std * (randn(size(rx_signal_clean)) + 1i*randn(size(rx_signal_clean)));
        
        %Receiver
        rx_grid = zeros(N_fft, num_total_sym);
        for sym_idx = 1:num_total_sym
            start_idx = (sym_idx-1)*(N_fft + N_cp) + N_cp + 1;
            end_idx = sym_idx*(N_fft + N_cp);
            rx_grid(:, sym_idx) = fft(rx_signal(start_idx:end_idx).', N_fft) / sqrt(N_fft);
        end
        
        %Channel Estimation
        rx_pilots = rx_grid(:, 1:num_pilot_sym);
        H_LS = zeros(M, N_fft);
        
        for tx_ant = 1:M
            % Despreading Walsh sequence to isolate the specific antenna
            decoupled_pilot = zeros(N_fft, 1);
            for sym_idx = 1:num_pilot_sym
                decoupled_pilot = decoupled_pilot + W(tx_ant, sym_idx) * rx_pilots(:, sym_idx);
            end
            decoupled_pilot = decoupled_pilot / num_pilot_sym;
            H_LS(tx_ant, :) = (decoupled_pilot ./ pilot_sym).';
        end
        
        H_MMSE = zeros(M, N_fft);
        for tx_ant = 1:M
            H_MMSE(tx_ant, :) = (W_MMSE * H_LS(tx_ant, :).').';
        end
        
        %Equalization & Demodulation
        rx_data = rx_grid(:, num_pilot_sym+1:end);
        
        %Calculating effective channel for the broadcasted data
        H_LS_comb = sum(H_LS, 1) / sqrt(M);
        H_MMSE_comb = sum(H_MMSE, 1) / sqrt(M);
        
        for d_idx = 1:num_data_sym
            Y_d = rx_data(:, d_idx).'; 
            
            X_est_LS = Y_d ./ H_LS_comb;
            X_est_MMSE = Y_d ./ H_MMSE_comb;
            
            rx_sym_ls = qamdemod(X_est_LS.', mod_order, 'UnitAveragePower', true);
            rx_sym_mmse = qamdemod(X_est_MMSE.', mod_order, 'UnitAveragePower', true);
            
            true_sym = squeeze(data_sym_raw(1, :, d_idx)).';
            
            err_ls = err_ls + sum(rx_sym_ls ~= true_sym);
            err_mmse = err_mmse + sum(rx_sym_mmse ~= true_sym);
        end
    end
    
    total_syms = num_iter * num_data_sym * N_fft;
    ser_ls(si) = err_ls / total_syms;
    ser_mmse(si) = err_mmse / total_syms;
    
    fprintf('SNR = %2d dB | LS SER = %.4f | MMSE SER = %.4f\n', snr_db, ser_ls(si), ser_mmse(si));
end

%% Plots
figure('Name', 'Part B Task 2: SER vs SNR');
semilogy(SNR_vec, ser_ls, 'r-o', 'LineWidth', 2, 'DisplayName', 'LS'); hold on;
semilogy(SNR_vec, ser_mmse, 'b-s', 'LineWidth', 2, 'DisplayName', 'MMSE');
grid on; xlabel('SNR (dB)'); ylabel('Symbol Error Rate (SER)');
title('RIS-Assisted MU-MISO: SER vs SNR');
legend('Location', 'southwest'); ylim([1e-4 1]);

