%% read_csi.m
%
% read CSI from UDPs created using the nexmon CSI extractor (nexmon.org/csi)
% modify the configuration section to your needs
% make sure you run >mex unpack_float.c before reading values from bcm4358 or bcm4366c0 for the first time
%
% the example.pcap file contains 4(core 0-1, nss 0-1) packets captured on a bcm4358
%
function [csi_real, csi_imag, rssi, sequence_nums, timestamps] = read_csi( ...
    file,        ...
    bandwidth,   ...
    num_cores,   ...
    num_streams, ...
    save_file    ...
    )
arguments
    file        (1, :)  char 
    bandwidth           uint32  {mustBeBandwidth} = 20
    num_cores           uint8                     = 1
    num_streams         uint8                     = 1
    save_file   (1, :)  char                      = ''
end

% unpack_float must be compiled with mex to run this script.
% As failsafe method, we explicitly check this here.
ensure_unpack_float_compiled()

% fft_size, which is the number of total subcarriers
fft_size = uint16(bandwidth * 3.2);

% read pcap file
pcap = readpcap();
pcap.open(file);

% We first sanity check whether the number of packets in this file makes sense
% CSI packets can become rather large in terms of payload, so they are fragmented
% across multiple UDP packets, one for each stream/antenna/wifipacket combination.
waiting_for_fragment = false;
num_udp_packets = length(pcap.all());
if rem(num_udp_packets, num_cores * num_streams) ~= 0
    error('Since we expect one packet per core (antenna) and stream, the total number should be a multiple!')
end
pcap.from_start();

% Perform efficient pre-allocation of buffers for the things we extract
num_captures  = num_udp_packets / (num_cores * num_streams);
csi           = complex(zeros(num_captures, num_cores, num_streams, fft_size), 0);
rssi          = zeros(num_captures, num_cores, 1);
sequence_nums = zeros(num_captures, 1);
timestamps    = zeros(num_captures, 1);

% Temporary antenna powers array. Antenna powers (RSSI) is extracted from only
% the first fragment of a bunch of CSI packets.
antenna_powers = zeros(num_cores, 1);

% Keep track of which wifi frame the current CSI was extracted from
currframe = 1;
while (true)
    frame = pcap.next();
    if isempty(frame)
        break;
    end

    if isempty(frame.payload)
	disp("Wtf empty frame payload?");
	break;
    end

    % Sanity check that this is ethernet + ip + udp
    ethertype = ntohs(frame.payload(13:14));
    ipv4      = frame.payload(15);
    protocol  = frame.payload(24);

    if ethertype ~= 2048 || ipv4 ~= 69 || protocol ~= 17
        disp ('Wrong packet in trace, skipping');
        continue;
    end

    % extract csi header
    csidata = frame.payload(43:end);

    if length(csidata) <= 44
        disp ('Packet is too short or contains only header, skipping');
        continue;
    end

    % Extract all the things present in the header!
    version             = csidata(1);             % version, at the moment must be zero
    num_extensions      = csidata(2);             % number of extensions that follow this header (zero if None)
    header_length       = ntohs(csidata(3:4));    % total length in byte of the header
    nexmon_chip         = ntohs(csidata(5:6));    % to identify the chip that extracted this data
    csi_data_format     = ntohs(csidata(7:8));    % format of the csi data or invalid and it requires an extension
    collectorchanwidth  = ntohs(csidata(9:10));   % spectral channel width configured on collector (20, 40, 80, 160)
    collectorchannumber = ntohs(csidata(11:12));  % channel number for central frequency (not primary channel)
    csichanwidth        = ntohs(csidata(13:14));  % width of the actual data that was captured
    csichannumber       = ntohs(csidata(15:16));  % channel number for central frequency of captured CSI data
    core_confmask       = ntohs(csidata(17:18));  % bitmask indicating on which core(s) the CSI was captured
    nss_confmask        = ntohs(csidata(19:20));  % bitmask indicating which nss(s) were to be captured
    sequence_number     = ntohl(csidata(21:24));  % sequence number of packet from which CSI was captured
    frametype           = csidata(25);            % type of frame that triggered CSI extraction
    frameencoding       = csidata(26);            % 0 -> non-HT, 1 -> HT, 2 -> VHT, 3 -> HE, 4 -> EHT ...
    framemcs            = csidata(27);            % mcs of the triggering frame
    framenss            = csidata(28);            % number of nss of frame that triggered the CSI extraction
    core                = csidata(29);            % core on which this CSI part was captured
    nss                 = csidata(30);            % spatial stream from which this CSI part was extracted
    total_subcarriers   = ntohs(csidata(31:32));  % number of subcarriers composing CSI profile for this core and nss
    start_subcarrier    = ntohs(csidata(33:34));  % first subc of this profile: data can fragment over UDP datagrams
    stop_subcarrier     = ntohs(csidata(35:36));  % last sybc of this profile: data can fragment over UDP datagrams
    csi_part_total      = ntohs(csidata(37:38));  % number of packets (parts) over which spans the current CSI
    csi_part_index      = ntohs(csidata(39:40));  % for fragment CSI: part indicator index
    mactime             = ntohl(csidata(41:44));  % time of d11 core in us when CSI collection was triggered

    % Extractor is capable of dealing with different bandwidths. However, for
    % the sake of our experiments, we want to catch issues early. Since we only
    % use one configured channel at a time, this must always be the same bw.
    if csichanwidth ~= bandwidth
        error('Captured CSI of invalid bandwidth found');
    end


    % process extensions
    if num_extensions > 0
        header_extension = csidata(45:header_length);
        if length(header_extension) < 4
            continue;
        end

        % we use this to extract multiple antenna extensions
        next_antenna = 0;

        for kk = 1:num_extensions
            % extract data for this extension
            extension_type = ntohs(header_extension(1:2));
            extension_length = ntohs(header_extension(3:4));
            extension_data = header_extension(5:extension_length);

            % check if powers are present
            if extension_type == 2
                % this extension is expected only in first part
                if csi_part_index > 0
                    disp 'Unexpected antenna extension in part > 0';
                end

                % cumulate antenna powers (more antenna extensions can be present)
                antenna_bitmask = ntohs(extension_data(1:2));
                power_data = extension_data(3:end);
                for antenna = 0:15
                    if bitand(antenna_bitmask, 2^antenna)
                        if antenna > num_cores + 1
                            error("Antenna reported RSSI even though core exceeds num configured cores")
                        end

                        antenna_powers(antenna + 1) = ntohs(power_data(1:2));

                        % extract data for next antenna
                        power_data = power_data(3:end);
                    end
                end
                next_antenna = next_antenna + 16;

                % check if frame payload is present
            elseif extension_type == 3
                % this extension is expected only in first part
                if csi_part_index > 0
                    disp 'Unexpected payload extension in part > 0';
                end
                frame.payload = dec2hex(extension_data(5:end));

                % if frame is long enough extract sequence counter
                if size(frame.payload, 1) >= 24 && frame.payload(1, 2) == '8'
                    tmp = ntohsswapped(extension_data(27:28));
                    sequence_nums(currframe) = tmp / 16;
                    %frame.fn = mod(tmp, 16);
                end
            end

            % extract data for next extension
            % this is possible also if extension was not processed
            header_extension = header_extension(extension_length + 1:end);
        end
    end

    if ~waiting_for_fragment
        % we are not waiting other parts, this must be first part of next csi
        if csi_part_index ~= 0
            disp('Part not expected, dropping...');
            continue;
        end

        % start a new csi
        next_part = 0;
        waiting_for_fragment = true;
        expected_sequence_number = sequence_number;

        % From the first fragment, we extract RSSI (from the corresponding extension)
        % as well as the timestamps from the pcap packet.
        rssi(currframe, :)  = antenna_powers;
        timestamps(currframe) = uint64(frame.header.ts_sec) * 1000000 + uint64(frame.header.ts_usec);
    end

    if csi_part_index ~= next_part
        disp 'Incomplete csi, dropping';
        waiting_for_fragment = false;
        continue;
    end

    if expected_sequence_number ~= sequence_number
        disp 'Incomplete csi(2), dropping';
        waiting_for_fragment = false;
        continue;
    end

    if start_subcarrier == 0
        % this is the first part for the csi of this core/nss, initialise
        csidata_core_nss = [];
    end

    csidata_core_nss = [csidata_core_nss; csidata(header_length + 1:end)];

    if stop_subcarrier + 1 == total_subcarriers
        % we are done with the csi for this core/nss, extract the full CSI
        csidata_core_nss_final = extract_csi_data( ...
            csidata_core_nss, ...
            csi_data_format,  ...
            frameencoding,    ...
            csichanwidth,     ...
            total_subcarriers ...
            );
        csi(currframe, core+1, nss+1, :) = csidata_core_nss_final.';
    end

    if csi_part_index == csi_part_total - 1
        % Here we are finished with all CSI extracted from this one WiFI frame
        % This means we may move on to the next one!
        currframe = currframe + 1;
        waiting_for_fragment = false;
        antenna_powers = zeros(num_cores, 1);

        continue;
    end

    % Per default, we have to wait for the next CSI packet segment
    next_part = csi_part_index + 1;
end

csi_real = real(csi);
csi_imag = imag(csi);

if (~isempty(save_file))
    save(save_file, 'timestamps', 'csi', 'rssi', 'sequence_nums', '-v7');
end
end


% ---------------------------------------------------------------------------
function csidata_core_nss_final = extract_csi_data(...
    csidata_core_nss, ...
    csi_data_format,  ...
    frameencoding,    ...
    csichanwidth,     ...
    total_subcarriers ...
    )
switch csi_data_format
    case 65535
        error('Unsupported csi data format extension, terminating');

    case 0
        % Convert to int16 and unpack
        csi_flat = typecast(csidata_core_nss, 'int16');
        csidata_core_nss_final = double(csi_flat(1:2:end)) + 1j * double(csi_flat(2:2:end));

        % Clean up based on channel width
        if frameencoding == 2
            switch csichanwidth
                case 80
                    idx = [1:6 128:130 252:256];
                case 40
                    idx = [1:6 64:66 124:128];
                case 20
                    idx = [1:4 33 62:64];
                otherwise
                    idx = []; % No cleaning for other widths
            end
            csidata_core_nss_final(idx) = 0;
        end

    case 1
        % Convert to uint32 and clean up before unpacking
        csi_flat = typecast(csidata_core_nss, 'uint32');

        if frameencoding == 3
            switch csichanwidth
                case 160
                    idx = mod(1024 + [0:11 510:514 1012:1023 1024:1035 1533:1538 2037:2047], 2048) + 1;
                case 80
                    idx = mod(512 + [0:2 501:511 512:523 1022:1023], 1024) + 1;
                case 40
                    idx = mod(256 + [0:2 245:255 256:267 510:511], 512) + 1;
                case 20
                    idx = mod(128 + [0:1 123:127 128:133 255], 256) + 1;
                otherwise
                    idx = []; % No cleaning for other widths
            end
            csi_flat(idx) = 0;
        end

        % Unpack and reconstruct the final result
        csi_flat_unpacked = unpack_float(int32(1), int32(1), int32(total_subcarriers), csi_flat)';
        csidata_core_nss_final = double(csi_flat_unpacked(1:2:end)) + 1j * double(csi_flat_unpacked(2:2:end));

    otherwise
        error('Unsupported csi data format %d, terminating', csi_data_format);
end
end


% ---------------------------------------------------------------------------
% Helper functions
% Argument validation function confirming that x is a valid bandwidth value.
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

% Function ensuring unpack_float was mex-compiled
function ensure_unpack_float_compiled()
% Ensure unpack_float was compiled
% NOTE: To allow for usage of this script outside of the current working directory,
% we manually build the full path to the unpack_float mexa file.
curr_file         = mfilename('fullpath');
[filepath, ~, ~]  = fileparts(curr_file);
unpack_float_file = fullfile(filepath, 'unpack_float.mexa64');
unpack_float_mac  = fullfile(filepath, 'unpack_float.mexmaca64');
if ~isfile(unpack_float_file) && ~isfile(unpack_float_mac)
    error('Please run `mex unpack_float.c` in the matlab directory first!');
end
end

% endianess conversion functions
function [val] = ntohs(bytes)
val = uint16(bytes(1)) * 256 + uint16(bytes(2));
end

function [val] = ntohsswapped(bytes)
val = uint16(bytes(2)) * 256 + uint16(bytes(1));
end

function [val] = ntohl(bytes)
val = uint32(bytes(1)) * 16777216 + uint32(bytes(2)) * 65536 + uint32(bytes(3)) * 256 + uint32(bytes(4));
end
