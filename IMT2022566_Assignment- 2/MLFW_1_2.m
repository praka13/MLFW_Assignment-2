clear; clc; close all;

%% Simulation Parameters
rng(42);                % Setting seed for reproducibility
N_fft = 64;             % Number of subcarriers
N_cp = 16;              % Cyclic Prefix Length
M = 4;                  % Modulation order (QPSK)
k = log2(M);            % Bits per symbol
SNR_vec = 0:5:30;       % SNR range in dB
num_iter = 1000;        % Monte Carlo iterations

% Pilot indices for the 8-pilot scenario
pilot_idx_8 = 1:8:64;   
num_pilots_8 = length(pilot_idx_8);
data_idx_8 = setdiff(1:N_fft, pilot_idx_8); 

%% Channel Statistics & Correlation Matrices
max_delay = 16;
tau = 0:max_delay-1;
pdp = exp(-tau/2);          % Exponential decay factor
pdp = pdp / sum(pdp);       % Normalizing power delay profile

% Theoretical MMSE Covariance Matrix (R_HH) based on PDP
R_HH = zeros(N_fft, N_fft);
for r = 1:N_fft
    for c = 1:N_fft
        val = 0;
        for l = 1:max_delay
            val = val + pdp(l) * exp(-1i*2*pi*(r-c)*(l-1)/N_fft);
        end
        R_HH(r,c) = val;
    end
end


R_Hp_Hp = R_HH(pilot_idx_8, pilot_idx_8);  % 8x8 matrix
R_Hall_Hp = R_HH(:, pilot_idx_8);          % 64x8 matrix

I_64 = eye(N_fft);
I_8 = eye(num_pilots_8);
E_x = 1; % Unit average power

%% Pre-allocating BER Arrays
ber_ls_64 = zeros(length(SNR_vec), 1);
ber_mmse_64 = zeros(length(SNR_vec), 1);
ber_ls_8 = zeros(length(SNR_vec), 1);
ber_mmse_8 = zeros(length(SNR_vec), 1);
ber_ls_no_cp = zeros(length(SNR_vec), 1);
ber_mmse_no_cp = zeros(length(SNR_vec), 1);

%% Main Simulation Loop

for snr_idx = 1:length(SNR_vec)
    snr_db = SNR_vec(snr_idx);
    snr_linear = 10^(snr_db/10);
    noise_var = 1 / snr_linear;
    
    % Compute MMSE Weight Matrices for this SNR
    W_MMSE_64 = R_HH / (R_HH + (noise_var / E_x) * I_64);
    W_MMSE_8  = R_Hall_Hp / (R_Hp_Hp + (noise_var / E_x) * I_8);
    
    % Error counters
    err_ls_64 = 0; err_mmse_64 = 0;
    err_ls_8 = 0;  err_mmse_8 = 0;
    err_ls_no_cp = 0; err_mmse_no_cp = 0;
    total_bits = 0;
    
    for iter = 1:num_iter
        
        %Generate Data & Modulate
        data_bits = randi([0 1], N_fft * k, 1);
        data_sym = qammod(data_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
        
        % Pilots for 64-pilot case
        pilot_bits_64 = randi([0 1], N_fft * k, 1);
        pilot_sym_64 = qammod(pilot_bits_64, M, 'InputType', 'bit', 'UnitAveragePower', true);
        
        % Pilots & Data for 8-pilot case
        sym_8_block = zeros(N_fft, 1);
        pilot_bits_8 = randi([0 1], num_pilots_8 * k, 1);
        pilot_sym_8 = qammod(pilot_bits_8, M, 'InputType', 'bit', 'UnitAveragePower', true);
        sym_8_block(pilot_idx_8) = pilot_sym_8;
        
        extra_data_bits = randi([0 1], length(data_idx_8) * k, 1);
        sym_8_block(data_idx_8) = qammod(extra_data_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
        
        %OFDM Modulation (IFFT) & CP Addition
        tx_pilot_time_64 = ifft(pilot_sym_64, N_fft) * sqrt(N_fft);
        tx_pilot_time_8  = ifft(sym_8_block, N_fft) * sqrt(N_fft);
        tx_data_time     = ifft(data_sym, N_fft) * sqrt(N_fft);
        
        tx_pilot_cp_64 = [tx_pilot_time_64(end-N_cp+1:end); tx_pilot_time_64];
        tx_pilot_cp_8  = [tx_pilot_time_8(end-N_cp+1:end); tx_pilot_time_8];
        tx_data_cp     = [tx_data_time(end-N_cp+1:end); tx_data_time];
        
        frame_64 = [tx_pilot_cp_64; tx_data_cp];
        frame_8  = [tx_pilot_cp_8; tx_data_cp];
        frame_no_cp = [tx_pilot_time_64; tx_data_time]; % Severe ISI
        
        %Channel & Noise Generation
        h_time = sqrt(pdp/2)' .* (randn(max_delay, 1) + 1i*randn(max_delay, 1));
        
        rx_clean_64    = filter(h_time, 1, frame_64);
        rx_clean_8     = filter(h_time, 1, frame_8);
        rx_clean_no_cp = filter(h_time, 1, frame_no_cp);
        
        noise_64    = sqrt(noise_var/2) * (randn(size(rx_clean_64)) + 1i*randn(size(rx_clean_64)));
        noise_8     = sqrt(noise_var/2) * (randn(size(rx_clean_8)) + 1i*randn(size(rx_clean_8)));
        noise_no_cp = sqrt(noise_var/2) * (randn(size(rx_clean_no_cp)) + 1i*randn(size(rx_clean_no_cp)));
        
        rx_sig_64    = rx_clean_64 + noise_64;
        rx_sig_8     = rx_clean_8 + noise_8;
        rx_sig_no_cp = rx_clean_no_cp + noise_no_cp;
        
        %Receiver Processing & Estimation
        
        %1: 64 Pilots (With CP)
        rx_pilot_64 = fft(rx_sig_64(N_cp+1 : N_fft+N_cp)) / sqrt(N_fft);
        rx_data_64  = fft(rx_sig_64(N_fft+2*N_cp+1 : end)) / sqrt(N_fft);
        
        H_LS_64 = rx_pilot_64 ./ pilot_sym_64;
        H_MMSE_64 = W_MMSE_64 * H_LS_64;
        
        bits_ls_64   = qamdemod(rx_data_64 ./ H_LS_64, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        bits_mmse_64 = qamdemod(rx_data_64 ./ H_MMSE_64, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        
        %8 Pilots (With CP)
        rx_pilot_8 = fft(rx_sig_8(N_cp+1 : N_fft+N_cp)) / sqrt(N_fft);
        rx_data_8  = fft(rx_sig_8(N_fft+2*N_cp+1 : end)) / sqrt(N_fft);
        
        H_LS_8_points = rx_pilot_8(pilot_idx_8) ./ pilot_sym_8;
        
       
        H_LS_8_interp = interp1(pilot_idx_8, H_LS_8_points, 1:N_fft, 'pchip', 'extrap').';
        H_MMSE_8 = W_MMSE_8 * H_LS_8_points; 
        
        bits_ls_8   = qamdemod(rx_data_8 ./ H_LS_8_interp, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        bits_mmse_8 = qamdemod(rx_data_8 ./ H_MMSE_8, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        
        %64 Pilots (No CP)
        rx_pilot_no_cp = fft(rx_sig_no_cp(1 : N_fft)) / sqrt(N_fft);
        rx_data_no_cp  = fft(rx_sig_no_cp(N_fft+1 : end)) / sqrt(N_fft);
        
        H_LS_no_cp = rx_pilot_no_cp ./ pilot_sym_64;
        H_MMSE_no_cp = W_MMSE_64 * H_LS_no_cp;
        
        bits_ls_no_cp   = qamdemod(rx_data_no_cp ./ H_LS_no_cp, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        bits_mmse_no_cp = qamdemod(rx_data_no_cp ./ H_MMSE_no_cp, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        
        
        err_ls_64 = err_ls_64 + biterr(data_bits, bits_ls_64);
        err_mmse_64 = err_mmse_64 + biterr(data_bits, bits_mmse_64);
        
        err_ls_8 = err_ls_8 + biterr(data_bits, bits_ls_8);
        err_mmse_8 = err_mmse_8 + biterr(data_bits, bits_mmse_8);
        
        err_ls_no_cp = err_ls_no_cp + biterr(data_bits, bits_ls_no_cp);
        err_mmse_no_cp = err_mmse_no_cp + biterr(data_bits, bits_mmse_no_cp);
        
        total_bits = total_bits + length(data_bits);
    end
    
    % Calculating Final BER for this SNR
    ber_ls_64(snr_idx) = err_ls_64 / total_bits;
    ber_mmse_64(snr_idx) = err_mmse_64 / total_bits;
    
    ber_ls_8(snr_idx) = err_ls_8 / total_bits;
    ber_mmse_8(snr_idx) = err_mmse_8 / total_bits;
    
    ber_ls_no_cp(snr_idx) = err_ls_no_cp / total_bits;
    ber_mmse_no_cp(snr_idx) = err_mmse_no_cp / total_bits;
    
    fprintf('SNR = %2d dB processed.\n', snr_db);
end

%% Plotting Results
% Fig1: Pilots = 64
figure('Name', 'Fig 1: Pilots = 64');
semilogy(SNR_vec, ber_ls_64, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNR_vec, ber_mmse_64, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('SNR (dB)', 'FontWeight', 'bold');
ylabel('BER', 'FontWeight', 'bold');
title('Fig.1: BER vs SNR (64 Pilots, With CP)');
legend('64 Pilots LS', '64 Pilots MMSE', 'Location', 'southwest');
ylim([1e-4 1]);

% Fig2: Pilots = 8
figure('Name', 'Fig 2: Pilots = 8');
semilogy(SNR_vec, ber_ls_8, 'm-^', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNR_vec, ber_mmse_8, 'k-d', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('SNR (dB)', 'FontWeight', 'bold');
ylabel('BER', 'FontWeight', 'bold');
title('Fig.2: BER vs SNR (8 Pilots, With CP)');
legend('8 Pilots LS (Interpolated)', '8 Pilots MMSE (Projected)', 'Location', 'southwest');
ylim([1e-4 1]);

% Fig3: With & Without CP for Pilots = 64
figure('Name', 'Fig 3: CP vs No CP');
semilogy(SNR_vec, ber_ls_64, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNR_vec, ber_mmse_64, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 7);
semilogy(SNR_vec, ber_ls_no_cp, 'g--o', 'LineWidth', 1.5, 'MarkerSize', 7);
semilogy(SNR_vec, ber_mmse_no_cp, 'c--s', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('SNR (dB)', 'FontWeight', 'bold');
ylabel('BER', 'FontWeight', 'bold');
title('Fig.3: Impact of CP Removal (64 Pilots)');
legend('LS With CP', 'MMSE With CP', 'LS Without CP', 'MMSE Without CP', 'Location', 'southwest');
ylim([1e-4 1]);

