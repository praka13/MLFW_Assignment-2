clear; clc; close all;

%% Common System Parameters
rng(42);                % seeding for reproducibility 
N_fft = 64;             % subcarriers
N_cp = 16;              % CP Length
M = 4;                  % QPSK
k = log2(M);            % bits per symbol
max_delay = 16;         % Maximum channel delay spread
snr_db = 15;            % Test SNR for the single run

% Noise setup
snr_linear = 10^(snr_db/10);
noise_var = 1 / snr_linear;
E_x = 1;                % Expected symbol energy 



%% WINNER II Channel Generation 
AA = winner2.AntennaArray('ULA', 1, 0.5); 

% 1 Base Station, 1 Mobile Station
MSIdx = [2];     % MS uses 2nd antenna array
BSIdx = {1};     % BS uses 1st antenna array 
cfgLayout = winner2.layoutparset(MSIdx, BSIdx, 1, [AA, AA]); 
cfgLayout.ScenarioVector = 3;        

% Configuring Model Parameters
cfgWim = winner2.wimparset;
cfgWim.CenterFrequency = 2.6e9;
cfgWim.RandomSeed = 42; 
cfgWim.NumTimeSamples = N_fft; 
sample_rate = 15e3 * N_fft; 
cfgWim.DelaySamplingInterval = 1 / sample_rate;

% Creating the built-in WINNER II Channel System Object
WINNER_chan = comm.WINNER2Channel(cfgWim, cfgLayout);

% Passing a Dirac impulse
impulse_in = complex([1; zeros(N_fft-1, 1)]);
h_raw_full = WINNER_chan(impulse_in);


if iscell(h_raw_full)
    h_raw_matrix = h_raw_full{1}; 
else
    h_raw_matrix = h_raw_full;    
end
h_raw = h_raw_matrix(:, 1);       

% Mapping to standard max_delay (16 taps) to act as our static filter
h_time = zeros(max_delay, 1);
num_taps = min(length(h_raw), max_delay);
h_time(1:num_taps) = h_raw(1:num_taps);

H_true_freq = fft(h_time, N_fft); 

% Generating discrete Power Delay Profile (PDP) for MMSE Covariance Matrix
pdp_discrete = abs(h_time).^2;
if sum(pdp_discrete) > 0
    pdp_discrete = pdp_discrete / sum(pdp_discrete);
end

% Computing theoretical R_HH for MMSE based on the exact built-in WINNER II PDP
R_HH_stat = zeros(N_fft, N_fft);
for r = 1:N_fft
    for c = 1:N_fft
        val = 0;
        for l = 1:max_delay
            val = val + pdp_discrete(l) * exp(-1i*2*pi*(r-c)*(l-1)/N_fft);
        end
        R_HH_stat(r,c) = val;
    end
end


% 64 PILOTS

fprintf('Testing Case A: 64 Pilots\n');


% MMSE Weight Matrix for 64 Pilots
W_MMSE_64 = R_HH_stat * inv(R_HH_stat + (noise_var / E_x) * eye(N_fft));

% Transmitter
pilot_bits_64 = randi([0 1], N_fft * k, 1);
data_bits_64 = randi([0 1], N_fft * k, 1);
pilot_sym_64 = qammod(pilot_bits_64, M, 'InputType', 'bit', 'UnitAveragePower', true);
data_sym_64 = qammod(data_bits_64, M, 'InputType', 'bit', 'UnitAveragePower', true);

tx_pilot_time_64 = ifft(pilot_sym_64, N_fft) * sqrt(N_fft);
tx_data_time_64 = ifft(data_sym_64, N_fft) * sqrt(N_fft);

tx_signal_64 = [tx_pilot_time_64(end-N_cp+1:end); tx_pilot_time_64; ...
                tx_data_time_64(end-N_cp+1:end); tx_data_time_64];

% Channel & Noise 
rx_clean_64 = filter(h_time, 1, tx_signal_64);
noise_64 = sqrt(noise_var/2) * (randn(size(rx_clean_64)) + 1i*randn(size(rx_clean_64)));
rx_signal_64 = rx_clean_64 + noise_64;

% Receiver
rx_pilot_freq_64 = fft(rx_signal_64(N_cp+1 : N_fft+N_cp), N_fft) / sqrt(N_fft);
rx_data_freq_64  = fft(rx_signal_64(N_fft+2*N_cp+1 : end), N_fft) / sqrt(N_fft);

% Estimation
H_LS_64 = rx_pilot_freq_64 ./ pilot_sym_64;
H_MMSE_64 = W_MMSE_64 * H_LS_64;

% Equalization & Verification
data_eq_MMSE_64 = rx_data_freq_64 ./ H_MMSE_64;
rx_bits_MMSE_64 = qamdemod(data_eq_MMSE_64, M, 'OutputType', 'bit', 'UnitAveragePower', true);
fprintf('64-Pilot Frame Processed. Bit Errors (MMSE): %d out of %d\n\n', biterr(data_bits_64, rx_bits_MMSE_64), length(data_bits_64));

% 8 PILOTS (Comb-type)

fprintf('Testing Case B: 8 Pilots\n');

num_pilots = 8;
pilot_spacing = N_fft / num_pilots;
pilot_idx = 1:pilot_spacing:N_fft; 

% Extracting cross-correlation matrices for 8-pilot MMSE
R_Hp_Hp = R_HH_stat(pilot_idx, pilot_idx); 
R_Hall_Hp = R_HH_stat(:, pilot_idx);       

% MMSE Weight Matrix for 8 Pilots -> 64 subcarriers
W_MMSE_8 = R_Hall_Hp * inv(R_Hp_Hp + (noise_var / E_x) * eye(num_pilots));

% Transmitter
pilot_bits_8 = randi([0 1], num_pilots * k, 1);
pilot_sym_8 = qammod(pilot_bits_8, M, 'InputType', 'bit', 'UnitAveragePower', true);

% Mapping 8 pilots to subcarriers, zero-pad the rest
tx_pilot_freq_8 = zeros(N_fft, 1);
tx_pilot_freq_8(pilot_idx) = pilot_sym_8;

data_bits_8 = randi([0 1], N_fft * k, 1);
data_sym_8 = qammod(data_bits_8, M, 'InputType', 'bit', 'UnitAveragePower', true);

tx_pilot_time_8 = ifft(tx_pilot_freq_8, N_fft) * sqrt(N_fft);
tx_data_time_8 = ifft(data_sym_8, N_fft) * sqrt(N_fft);

tx_signal_8 = [tx_pilot_time_8(end-N_cp+1:end); tx_pilot_time_8; ...
               tx_data_time_8(end-N_cp+1:end); tx_data_time_8];

% Channel 
rx_clean_8 = filter(h_time, 1, tx_signal_8);
noise_8 = sqrt(noise_var/2) * (randn(size(rx_clean_8)) + 1i*randn(size(rx_clean_8)));
rx_signal_8 = rx_clean_8 + noise_8;

% Receiver
rx_pilot_freq_block_8 = fft(rx_signal_8(N_cp+1 : N_fft+N_cp), N_fft) / sqrt(N_fft);
rx_data_freq_8  = fft(rx_signal_8(N_fft+2*N_cp+1 : end), N_fft) / sqrt(N_fft);

% Estimation
rx_pilots_only_8 = rx_pilot_freq_block_8(pilot_idx);
H_LS_pilots_8 = rx_pilots_only_8 ./ pilot_sym_8;

% LS Interpolation & MMSE Projection
H_LS_8_full = interp1(pilot_idx, H_LS_pilots_8, 1:N_fft, 'pchip', 'extrap').';
H_MMSE_8_full = W_MMSE_8 * H_LS_pilots_8;

% Equalization & Verification
data_eq_MMSE_8 = rx_data_freq_8 ./ H_MMSE_8_full;
rx_bits_MMSE_8 = qamdemod(data_eq_MMSE_8, M, 'OutputType', 'bit', 'UnitAveragePower', true);
fprintf('8-Pilot Frame Processed. Bit Errors (MMSE): %d out of %d\n\n', biterr(data_bits_8, rx_bits_MMSE_8), length(data_bits_8));




figure('Name', 'Task 1: Channel Estimation Comparison', 'Position', [100, 100, 1200, 500]);

%64 Pilots
subplot(1, 2, 1);
plot(1:N_fft, abs(H_true_freq), 'k-', 'LineWidth', 2); hold on;
plot(1:N_fft, abs(H_LS_64), 'ro--', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(1:N_fft, abs(H_MMSE_64), 'bx-.', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
title('64 Pilots: WINNER II Channel Response', 'FontWeight', 'bold');
xlabel('Subcarrier Index', 'FontWeight', 'bold');
ylabel('Magnitude |H|', 'FontWeight', 'bold');
legend('True Channel', 'LS Estimate', 'MMSE Estimate', 'Location', 'best');
xlim([1 64]);

% Pilots
subplot(1, 2, 2);
plot(1:N_fft, abs(H_true_freq), 'k-', 'LineWidth', 2); hold on;
plot(1:N_fft, abs(H_LS_8_full), 'r^--', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(1:N_fft, abs(H_MMSE_8_full), 'bd-.', 'LineWidth', 1.5, 'MarkerSize', 6);

plot(pilot_idx, abs(H_true_freq(pilot_idx)), 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
grid on;
title('8 Pilots: WINNER II Channel Response', 'FontWeight', 'bold');
xlabel('Subcarrier Index', 'FontWeight', 'bold');
ylabel('Magnitude |H|', 'FontWeight', 'bold');
legend('True Channel', 'LS (Interpolated)', 'MMSE (Projected)', 'Pilot Locations', 'Location', 'best');
xlim([1 64]);

