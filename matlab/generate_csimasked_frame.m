function [] = generate_csimasked_frame(  ... % Generate a range of frames with increasing sequence number
    outfile,             ...  % File in which to save samples
    sample_max_scale,    ...  % Scale value of waveform samples when writing to file (max value to scale to)
    receiver_address,    ...  % Receiver address for MAC header
    transmitter_address, ...  % Transmitter address for MAC header
    bssid_address,       ...  % BSSID address for MAC header
    bandwidth_mhz,       ...  % Bandwidth in MHz of frame to generate
    num_repeats,         ...  % Number of times to repeat masked frame sequence
    enable_sounding,     ...  % Whether to enable sounding bit in PHY preamble
    csi_precoding_masks, ...  % Mask(s) to use for precoding all IQ symbols in frame
    guard_interval_type, ...  % Type of guard interval; Either Long or Short.
    data_rate_mode       ...  % Whether to use very high throughput mode
    )
arguments
    outfile             (1, :)  char
    sample_max_scale            int16
    receiver_address    (1, 12) char
    transmitter_address (1, 12) char 
    bssid_address       (1, 12) char                      = transmitter_address
    bandwidth_mhz               uint32  {mustBeBandwidth} = 20
    num_repeats                 uint32  {mustBePositive}  = 1
    enable_sounding             logical                   = false
    csi_precoding_masks (:, :)          {mustBeNumeric}   = complex(ones([(bandwidth_mhz / 20) * 64 1]))
    guard_interval_type (1, :)  char                      = 'Short'
    data_rate_mode      (1, :)  char                      = 'Non-HT'
end

if strcmp(data_rate_mode, 'Non-HT')
    guard_interval_type = 'Long';
end

% Unpack and check precoding mask dimensions
[num_scs, num_masks] = size(csi_precoding_masks);
assert(num_scs == (bandwidth_mhz/20)*64, "Mask first dimension did not match number of subcarriers as per specified bandwidth.")

% Extract some relevant info about symble durations etc. for masking
masking_info = get_masking_info(bandwidth_mhz, guard_interval_type, data_rate_mode);

% Create phy and mac configuration from specified parameters.
% Initial frame always starts with sequence number being zero.
[phy_config, mac_config, payload] = create_frame_configs( ...
    receiver_address,    ...
    transmitter_address, ...
    bssid_address,       ...
    bandwidth_mhz,       ...
    0,                   ...
    enable_sounding,     ...
    guard_interval_type, ...
    data_rate_mode       ...
    );


% Creating initial waveform and mask it
% NOTE: We rescale before masking to ensure that gain changes in received frame
% are relative to the unmasked frame
initial_mask = csi_precoding_masks(:, 1);
waveform = create_frame_waveform(phy_config, mac_config, payload);
max_abs = max(abs(waveform));
waveform = rescale(waveform, sample_max_scale);
waveform = apply_mask_to_frame(waveform, initial_mask, masking_info);

fprintf(['Normalized waveform samples. \n' ...
    '  -- Maximum IQ amplitude    : %d \n' ...
    '  -- Maximum IQ after scaling: %d \n' ...
    '  -- Scale factor w.r.t. max : %d \n'], max_abs, max(abs(waveform)), sample_max_scale)

% Calculate expected sizes to allow preallocation of buffer for final
% sequence of frames
% NOTE:
%   - We generate groups of frames
%   - One group per mask (csi_precoding_masks may contain multiple masks)
%   - Every group is repeated num_repeats times back-to-back
num_frames = num_masks * num_repeats;
num_samples_per_frame = length(waveform);

% Preallocate full buffer for frame and populate with initial frame
fprintf(['Allocating buffer for file containing IQ samples. \n' ...
    '  -- Number of frames in file      : %d \n' ...
    '  -- Num complex samples per frame : %d \n'], num_frames, num_samples_per_frame)

waveforms = zeros(num_frames, num_samples_per_frame, 'like', waveform);
waveforms(1, :) = waveform;

for frame_num = 2:num_frames
    % Update sequence number
    % NOTE: Sequence number is a 10 bit MAC-layer primitive -> Maximum value 4095
    sequence_num = mod(frame_num - 1, 4096);
    mac_config.SequenceNumber = sequence_num;

    % We cycle through the masks as per repetitions. In other words, we generate a frame for
    % each mask and repeat that num_repeats amount of times. In terms of the total number of
    % frame repetitions, we can just use modulo here to achieve that.
    mask = csi_precoding_masks(:, mod(frame_num, num_masks) + 1);

    % Use configuration to create actual waveform samples and pad them
    waveform = create_frame_waveform(phy_config, mac_config, payload);
    waveform = rescale(waveform, sample_max_scale);
    waveform = apply_mask_to_frame(waveform, mask, masking_info);
    waveforms(frame_num, :) = waveform;
end

save(outfile, 'waveforms', '-v7');
end


% -------------------------------------------------------------------------------------------------
% Function to rescale waveform samples to fit within [-sample_max_scale, sample_max_scale] per
% IQ sample value.
% -------------------------------------------------------------------------------------------------
function [norm_samples] = rescale(wave_samples, sample_max_scale)
    max_abs = max(abs(wave_samples));
    norm_samples = wave_samples / max_abs * double(sample_max_scale);
end


% -------------------------------------------------------------------------------------------------
% Generate PHY and MAC layer configurations for the frame to generate
% -------------------------------------------------------------------------------------------------
function [phy_config, mac_config, payload] = create_frame_configs(...
    receiver_address,    ...
    transmitter_address, ...
    bssid_address,       ...
    bandwidth_mhz,       ...
    sequence_number,     ...
    enable_sounding,     ...
    guard_interval_type, ...
    data_rate_mode       ...
    )
arguments
    receiver_address    (1, 12) char
    transmitter_address (1, 12) char
    bssid_address       (1, 12) char    = transmitter_address
    bandwidth_mhz               uint32  = 20
    sequence_number             uint16  = 0
    enable_sounding             logical = false
    guard_interval_type (1, :)  char    = 'Short'
    data_rate_mode      (1, :)  char    = 'Non-HT'
end

% Choose correct Config and frame format depending on whether vht is chosen or not
% Figure out time a data symbol takes and the respective guard interval takes (in seconds)
if strcmp(data_rate_mode, 'Non-HT')
    phy_config = wlanNonHTConfig;
    frame_format = 'Non-HT';
elseif strcmp(data_rate_mode, 'HT')
    phy_config = wlanHTConfig;
    frame_format = 'HT-Mixed';
    phy_config.RecommendSmoothing  = 1;
    phy_config.ForceSounding       = enable_sounding;
    phy_config.NumSpaceTimeStreams = 1;
    phy_config.ChannelCoding       = 'BCC';
    phy_config.GuardInterval       = guard_interval_type;
elseif strcmp(data_rate_mode, 'VHT')
    phy_config = wlanVHTConfig;
    frame_format = 'VHT';
    phy_config.NumSpaceTimeStreams = 1;
    phy_config.ChannelCoding       = 'BCC';
    phy_config.GuardInterval       = guard_interval_type;
    phy_config.Beamforming         = false;
else
    msg = 'Given data rate mode not yet supported';
    error(msg);
end

% Physical layer configuration
phy_config.ChannelBandwidth    = sprintf('CBW%d', bandwidth_mhz);
phy_config.NumTransmitAntennas = 1;
phy_config.MCS                 = 0;

% MAC layer configuration
mac_config                 = wlanMACFrameConfig('FrameType', 'QoS Data');
mac_config.FrameFormat     = frame_format;
mac_config.ToDS            = 0;
mac_config.FromDS          = 1;
mac_config.AckPolicy       = 'Normal Ack/Implicit Block Ack Request';
mac_config.MSDUAggregation = false;
mac_config.MPDUAggregation = false;
mac_config.Address1        = receiver_address;
mac_config.Address2        = transmitter_address;
mac_config.Address3        = bssid_address;
mac_config.SequenceNumber  = sequence_number;

% Payload (this is a dummy payload and uninteresting for the experiments)
% This specific one is some random payload from a frame captured before.
payload = {'aaaa0300000008004500005451a44000400152b6c0a80ac7c0a80a3708005fa11dc1017aa214f163000000001cd80a0000000000101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031323334353637'};
end


% -------------------------------------------------------------------------------------------------
% Generate waveform samples from PHY and MAC layer configuration
% -------------------------------------------------------------------------------------------------
function waveform = create_frame_waveform(...
    phy_config, ...
    mac_config, ...
    payload     ...
    )
[frame_bits, frame_length] = wlanMACFrame(payload, mac_config, phy_config, 'OutputFormat', 'bits');

if strcmp(mac_config.FrameFormat, 'HT-Mixed')
    phy_config.PSDULength = frame_length;
elseif strcmp(mac_config.FrameFormat, 'VHT')
    phy_config.APEPLength = frame_length;
end

% Generate baseband packets
scrambler_initialization = 7;
waveform = wlanWaveformGenerator(...
    frame_bits, ...
    phy_config, ...
    'NumPackets', 1, ...
    'IdleTime', 0, ...
    'ScramblerInitialization', scrambler_initialization ...
    );
end


% -------------------------------------------------------------------------------------------------
% Aggregate phy header info relevant for masking
%
% Returns:
% struct WindowInfo:
%   window (int)                  : Window sample size
%   header_sample_sizes (list)    : List of header fields sample sizes
%   windowing_sample_offset (list): List of offset to skip for symbol extraction (e.g. skipping
%                                   guard interval) of symbol unaffected by window
% -------------------------------------------------------------------------------------------------
function [masking_info] = get_masking_info(...
    bandwidth_mhz,       ... % Bandwidth in MHz
    guard_interval_type, ... % Short or long
    data_rate_mode       ... % Whether to use VHT
    )
arguments
    bandwidth_mhz               uint32
    guard_interval_type (1, :)  char
    data_rate_mode      (1, :)  char
end

vht = strcmp(data_rate_mode, 'VHT');

masking_info.window = uint32(3);
if vht && bandwidth_mhz > 80
    masking_info.window = uint32(7);
end

% Figure out time a data symbol takes and the respective guard interval takes (in seconds)
if strcmp(guard_interval_type, 'Long')
    data_gi_length = 0.8;
elseif strcmp(guard_interval_type, 'Short')
    data_gi_length = 0.4;
else
    msg = 'Guard interval has unknown format -- Aborting';
    error(msg);
end

% NOTE: The actual data symbol is always 3.2us. The duration of the whole field changes
% depending on the guard interval length
data_symbol_time_ns = 3.2;
data_field_time_ns  = data_symbol_time_ns + data_gi_length;

% Non-HT header is composed of:
% 1           | 2           | 3
% L-STF (8us) | L-LTF (8us) | L-SIG (4us)
%
% HT header is composed by a sequence of pieces (SIG-A is actually two symbols)
%
% 1           | 2           | 3           | 4              | 5              | 6            | 7
% L-STF (8us) | L-LTF (8us) | L-SIG (4us) | HT-SIG-1 (4us) | HT-SIG-2 (4us) | HT-STF (4us) | HT-LTF (4us * stream)
%
% we have only one spatial stream so we have only one HT-LTF
%----------------------------------------------------------------------------------------------------------------------
% VHT header is composed by a sequence of pieces (SIG-A is actually two symbols)
%
% 1           | 2           | 3           | 4(A1)  | 5(A2)  | 6             | 7                      | 8
% L-STF (8us) | L-LTF (8us) | L-SIG (4us) | VHT-SIG-A (8us) | VHT-STF (4us) | VHT-LTF (4us * stream) | VHT-SIG-B (4us)
%


% we have only one spatial stream so we have only one VHT-LTF
% NOTE: Between Non-HT, HT and VHT, the headers just extend. The first three symbols
% are common to all.

% As described above, VHT includes an additional field
if strcmp(data_rate_mode, 'Non-HT')
    header_durations = [8 8   4 ];
    gi_sym_reps      = [1 0.5 0.25];
elseif strcmp(data_rate_mode, 'HT')
    header_durations = [8 8   4    4    4    4    4];
    gi_sym_reps      = [1 0.5 0.25 0.25 0.25 0.25 0.25];
elseif strcmp(data_rate_mode, 'VHT')
    header_durations = [8 8   4    4    4    4    4    4];
    gi_sym_reps      = [1 0.5 0.25 0.25 0.25 0.25 0.25 0.25];
end

% number of samples per symbol
symbol_length              = bandwidth_mhz / 20 * 64;
masking_info.symbol_length = symbol_length;

% Store sample sizes of header and interval to skip for windowing removal
masking_info.header_sample_sizes      = uint32(header_durations .* double(bandwidth_mhz));   % number of samples in header symbols
masking_info.extraction_sample_offset = uint32(gi_sym_reps      .* double(symbol_length));   % number of samples to skip for guard iv

% Store sample sizes of guard interval and data symbols for payload
masking_info.n_samples_per_data_gi    = uint32(data_gi_length     * bandwidth_mhz);
masking_info.n_samples_per_data_field = uint32(data_field_time_ns * bandwidth_mhz);
end


% -------------------------------------------------------------------------------------------------
% Apply mask to frame, i.e. precode all symbols in the frame.
% NOTE: Also removes windowing of symbols.
% -------------------------------------------------------------------------------------------------
function waveform = apply_mask_to_frame(...
    waveform,            ...
    csi_precoding_mask,  ...
    masking_info         ...
    )
% Alias unwrapping for better readability
window_len               = masking_info.window;                     % Window rolloff size in which samples are affected
symbol_length            = masking_info.symbol_length;              % number of samples per symbol without GI
header_sample_sizes      = masking_info.header_sample_sizes;        % Sample size of header fields
extraction_sample_offset = masking_info.extraction_sample_offset;   % Offset to skip for extraction of unaffected symbol
n_samples_per_gi         = masking_info.n_samples_per_data_gi;      % Data field guard interval sample size
n_samples_per_df         = masking_info.n_samples_per_data_field;   % Number of samples in data field
half_symbol_length       = symbol_length / 2;                       % Store to avoid frequent recomputation

% Step through the different PHY preamble symbols. For each symbol, remove
% windowing that appears at boundary samples by appropriately cutting from
% and unwindowed position and unrolling.
cursample = 0;
for headjj = 1 : length(header_sample_sizes)

    % Get number of samples and the interval to skip for extraction
    % NOTE: for all but the first field, the sample offset is simply the guard interval length.
    n_samples      = header_sample_sizes(headjj);
    sample_offset  = extraction_sample_offset(headjj);

    % extract field section  and single symbol unaffected by windowing
    sym_field = waveform(cursample + (1 : n_samples));
    symbol    = sym_field((sample_offset + 1) : (sample_offset + symbol_length));

    % Apply mask to unaffected symbol and reassemble corresponding field
    if headjj == 1
        symbol = apply_mask_to_symbol(symbol, csi_precoding_mask);
        cyc_postfix = symbol(1 : half_symbol_length);
        sym_field = [symbol; symbol; cyc_postfix];
    elseif headjj == 2
        symbol = apply_mask_to_symbol(symbol, csi_precoding_mask);
        cyc_prefix = symbol((end - half_symbol_length + 1) : end);
        sym_field = [cyc_prefix; symbol; symbol];
    elseif ismember (headjj, [3 4 5 6 7])
        % In these single parts of the preamble, we have only one symbol and cyclic
        % prefix. This means that the symbol (at the end) is affected by windowing.
        % We take the unaffected samples from the prefix to remove that windowing.
        guard_interval = sym_field(1 : sample_offset);
        symbol(end - window_len : end) = guard_interval(end - window_len : end);

        symbol = apply_mask_to_symbol(symbol, csi_precoding_mask);
        cyc_prefix = symbol(end - sample_offset + 1 : end);
        sym_field = [cyc_prefix; symbol];
    end

    waveform(cursample + (1 : n_samples)) = sym_field;
    cursample = cursample + n_samples;
end



% Sanity check for correct number of samples present
num_data_samples = (length(waveform) - cursample);
if mod(num_data_samples, n_samples_per_df) > 0
    msg = 'Invalid number of samples';
    error(msg);
end


% Run through data symbols, remove windowing and apply mask again
num_data_symbols = num_data_samples / n_samples_per_df;
for datajj = 1:num_data_symbols
    sym_field = waveform(cursample + (1 : n_samples_per_df));
    symbol = sym_field(n_samples_per_gi + 1 : end);

    % remove windowing at the end by taking samples from cyclic prefix
    symbol(end - window_len : end) = sym_field(n_samples_per_gi - window_len : n_samples_per_gi);

    % Mask and reassemble symbol samples
    symbol = apply_mask_to_symbol(symbol, csi_precoding_mask);
    cyc_prefix = symbol(end - n_samples_per_gi + 1 : end);
    waveform(cursample + (1 : n_samples_per_df)) = [cyc_prefix; symbol];

    cursample = cursample + n_samples_per_df;
end
end


% -------------------------------------------------------------------------------------------------
% Apply CSI precoding mask to a single symbol
% -------------------------------------------------------------------------------------------------
function symbol = apply_mask_to_symbol(symbol, mask)
symbol = fftshift(fft(symbol));
symbol = symbol .* mask;
symbol = ifft(fftshift(symbol));
end


% Argument validation function confirming that x is a type
% using complex storage.
function mustBeComplex(x)
% isreal is false even if imaginary part is zero
if isreal(x)
    eid = 'Type:NotComplex';
    msg = 'Mask must be all complex values.';
    throwAsCaller(MException(eid, msg))
end
end


% Argument validation function confirming that x is a type
% using complex storage.
function mustBeBandwidth(x)
x = uint32(x);
valid_bandwidths = [20, 40, 80, 160];
if ~ismember(x, valid_bandwidths)
    bw_str = strjoin(string(valid_bandwidths), ', ');
    val_str = string(x);
    eid = 'Type:NotValidBandwidth';
    msg = append('Invalid bandwidth value encountered.', ...
        '\n -- Value          : ', val_str, ...
        '\n -- Allowed values : ', bw_str   ...
        );
    disp(msg)
    throwAsCaller(MException(eid, msg))
end
end

