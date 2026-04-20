
clear; clc; close all;

%% System Parameters 
rng(42); % Setting seed for reproducibility
N_fft = 64;             % Number of subcarriers
N_cp = 16;              % Cyclic Prefix length
M = 4;                  % Number of BS antennas
K = 1;                  % Number of users
N_x = 16; N_y = 16;     % RIS grid dimensions
N = N_x * N_y;          % Total RIS elements (256)
L = 20;                 % Number of mmWave channel paths
f_c = 28e9;             % Carrier frequency = 28 GHz
c = 3e8;                % Speed of light
lambda = c / f_c;       % Wavelength
d = lambda / 2;         % Antenna element spacing
mod_order = 4;          % QPSK
k_bits = log2(mod_order);



%% mmWave Cascaded Channel Model Generation (3D Saleh-Valenzuela)


% Setting random phase shifts for all RIS elements
omega = rand(N, 1) * 2 * pi;
Phi = diag(exp(1i * omega));

% Channel 1: BS to RIS (Size: N x M)
alpha_1 = (randn(L, 1) + 1i*randn(L, 1)) / sqrt(2); % Complex path gains
tau_1 = randi([0, 7], L, 1); % Delays in samples
phi_r_1 = rand(L, 1) * pi; theta_r_1 = rand(L, 1) * pi; % AOA/ZOA at RIS
phi_t_1 = rand(L, 1) * pi; % AOD at BS

% Channel 2: RIS to UE (Size: 1 x N)
alpha_2 = (randn(L, 1) + 1i*randn(L, 1)) / sqrt(2);
tau_2 = randi([0, 8], L, 1); % Delays in samples
phi_t_2 = rand(L, 1) * pi; theta_t_2 = rand(L, 1) * pi; % AOD/ZOD at RIS

% function for BS Uniform Linear Array (ULA) response
a_BS = @(phi) (1/sqrt(M)) * exp(1i * pi * (0:M-1)' * sin(phi));

% Pre-computing RIS Uniform Planar Array (UPA) responses
a_RIS_H1 = zeros(N, L); 
a_RIS_H2 = zeros(N, L); 
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
a_RIS_H1 = (1/sqrt(N)) * a_RIS_H1;
a_RIS_H2 = (1/sqrt(N)) * a_RIS_H2;

% Building Frequency Domain Channels for all 64 subcarriers
H1_freq = zeros(N, M, N_fft);
H2_freq = zeros(1, N, N_fft);
for k_sub = 0:N_fft-1
    H1_k = zeros(N, M);
    H2_k = zeros(1, N);
    for l = 1:L
        % Summing the multipath contributions with delay phase shifts
        H1_k = H1_k + alpha_1(l) * a_RIS_H1(:, l) * a_BS(phi_t_1(l))' * exp(-1i * 2 * pi * k_sub * tau_1(l) / N_fft);
        H2_k = H2_k + alpha_2(l) * a_RIS_H2(:, l)' * exp(-1i * 2 * pi * k_sub * tau_2(l) / N_fft);
    end
    H1_freq(:, :, k_sub+1) = sqrt(M*N/L) * H1_k;
    H2_freq(1, :, k_sub+1) = sqrt(N/L) * H2_k;
end

% Computing Effective Cascaded Channel
Heff_freq = zeros(1, M, N_fft);
for k_sub = 1:N_fft
    Heff_freq(1, :, k_sub) = H2_freq(1, :, k_sub) * Phi * H1_freq(:, :, k_sub);
end

% Converting this to Time-Domain impulse response for accurate physical filtering
Heff_time = ifft(Heff_freq, N_fft, 3); 
Heff_time = Heff_time(:, :, 1:N_cp); % Truncate to CP length

%% OFDM Transmitter (BS)
fprintf('Building Transmitter and Mapping Pilots...\n');
num_pilot_sym = M; % 4 orthogonal pilot symbols required to estimate 4 antennas
num_data_sym = 1;

% Generating Base Pilot Sequence
pilot_base_bits = randi([0 1], N_fft*k_bits, 1);
pilot_base_sym = qammod(pilot_base_bits, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);

% Applying Walsh-Hadamard Matrix to make pilots orthogonal across the 4 antennas
W = hadamard(M); 
tx_grid_pilot = zeros(M, N_fft, num_pilot_sym);
for sym_idx = 1:num_pilot_sym
    for tx_ant = 1:M
        tx_grid_pilot(tx_ant, :, sym_idx) = W(tx_ant, sym_idx) * pilot_base_sym.';
    end
end

% Generate random QPSK Data
data_bits = randi([0 1], M * N_fft * k_bits, 1);
data_sym = qammod(data_bits, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);
tx_grid_data = reshape(data_sym, M, N_fft, 1);

% Combining Frame: [4 Pilot Symbols, 1 Data Symbol]
tx_grid = cat(3, tx_grid_pilot, tx_grid_data);
num_total_sym = size(tx_grid, 3);

% IFFT and CP Addition for each antenna
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

%% Channel Propagation & Interference
fprintf('Transmitting signal through Time-Domain Channel\n');
rx_signal_clean = zeros(1, size(tx_signal, 2));

% Applying specific time-domain filter for each transmit antenna path
for tx_ant = 1:M
    h_tap = squeeze(Heff_time(1, tx_ant, :)); 
    rx_signal_clean = rx_signal_clean + filter(h_tap, 1, tx_signal(tx_ant, :));
end

% Adding AWGN
SNR_dB = 20;
noise_var = 1 / (10^(SNR_dB/10));
noise = sqrt(noise_var/2) * (randn(size(rx_signal_clean)) + 1i*randn(size(rx_signal_clean)));
rx_signal = rx_signal_clean + noise;

%% OFDM Receiver 
fprintf('Receiver Processing (CP Removal & FFT)\n');
rx_grid = zeros(N_fft, num_total_sym);

for sym_idx = 1:num_total_sym
    start_idx = (sym_idx-1)*(N_fft + N_cp) + N_cp + 1;
    end_idx = sym_idx*(N_fft + N_cp);
    
    % CP removal and FFT to recover symbols
    sym_time = rx_signal(start_idx:end_idx).';
    rx_grid(:, sym_idx) = fft(sym_time, N_fft) / sqrt(N_fft);
end



%% Validation Plot
figure('Name', 'Task 1: mmWave Channel Validation');
plot(1:N_fft, 10*log10(abs(squeeze(Heff_freq(1,1,:)))), 'b-o', 'LineWidth', 1.5);
title('Frequency Response of Cascaded RIS Channel (BS Antenna 1 to UE)');
xlabel('Subcarrier Index'); ylabel('Magnitude (dB)');
grid on;