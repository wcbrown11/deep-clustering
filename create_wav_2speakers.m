% create_wav_2_speakers.m
%
% Create 2-speaker mixtures
% 
% This script assumes that WSJ0's wv1 sphere files have already
% been converted to wav files, using the original folder structure
% under wsj0/, e.g., 
% 11-1.1/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and 
% stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav, and
% 11-6.1/wsj0/si_dt_05/050/050a0501.wv1 is converted to wav and
% stored in YOUR_PATH/wsj0/si_dt_05/050/050a0501.wav.
% Relevant data from all disks are assumed merged under YOUR_PATH/wsj0/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2016 Mitsubishi Electric Research Labs 
%                          (Jonathan Le Roux, John R. Hershey, Zhuo Chen)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath('./voicebox')
data_type = {'tr','cv','tt'};
wsj0root = '/scratch/near/'; % YOUR_PATH/, the folder containing wsj0/
output_dir16k='/scratch/near/2speakers_6channel/wav16k';
output_dir8k='/scratch/near/2speakers_6channel/wav8k';

min_max = {'min'}; % {'min','max'};


for i_mm = 1:length(min_max)
    for i_type = 1:length(data_type)
        if ~exist([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        if ~exist([output_dir8k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir8k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/mix/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/']);
                
        TaskFile = ['mix_2_spk_' data_type{i_type} '.txt'];
        fid=fopen(TaskFile,'r');
        C=textscan(fid,'%s %f %s %f');
        
        Source1File = ['mix_2_spk_' min_max{i_mm} '_' data_type{i_type} '_1'];
        Source2File = ['mix_2_spk_' min_max{i_mm} '_' data_type{i_type} '_2'];
        MixFile     = ['mix_2_spk_' min_max{i_mm} '_' data_type{i_type} '_mix'];
        fid_s1 = fopen(Source1File,'w');
        fid_s2 = fopen(Source2File,'w');
        fid_m  = fopen(MixFile,'w');
        
        num_files = length(C{1});
        fs8k=8000;
        
        scaling_16k = zeros(num_files,2);
        scaling_8k = zeros(num_files,2);
        scaling16bit_16k = zeros(num_files,1);
        scaling16bit_8k = zeros(num_files,1);
        fprintf(1,'%s\n',[min_max{i_mm} '_' data_type{i_type}]);
        for i = 1:num_files
            [inwav1_dir,invwav1_name,inwav1_ext] = fileparts(C{1}{i});
            [inwav2_dir,invwav2_name,inwav2_ext] = fileparts(C{3}{i});
            fprintf(fid_s1,'%s\n',C{1}{i});
            fprintf(fid_s2,'%s\n',C{3}{i});
            inwav1_snr = C{2}(i);
            inwav2_snr = C{4}(i);
            mix_name = [invwav1_name,'_',num2str(inwav1_snr),'_',invwav2_name,'_',num2str(inwav2_snr)];
            fprintf(fid_m,'%s\n',mix_name);
                        
            % get input wavs
            [s1, fs] = audioread([wsj0root C{1}{i}]);
            s2       = audioread([wsj0root C{3}{i}]);
            % get impulse responses
            n = randi(500);
            file_ir = strcat('/scratch/near/roomGeometry/roomGeometry_',sprintf('%03d', n),'.mat');
            %ir2 = strcat('/scratch/near/roomGeometry/roomGeometry_',sprintf('%03d', n2),'.wav');
            ir = load(file_ir);
            ir = ir.result;
            ir = ir.';
            %all(ir(:,1)==ir(:,2))
            impulse_response_1 = ir(:,1:6);
            impulse_response_2 = ir(:,7:12);
            rever_1 = zeros(length(s1)+length(impulse_response_1(:,1))-1,6);
            rever_2 = zeros(length(s2)+length(impulse_response_2(:,1))-1,6);
            for channel = 1:6
                rever_1(:,channel) = fastconv(s1,impulse_response_1(:,channel));
                rever_2(:,channel) = fastconv(s2,impulse_response_2(:,channel));
            end
            s1 = rever_1;
            s2 = rever_2;

            % resample, normalize 8 kHz file, save scaling factor
            s1_8k=resample(s1,fs8k,fs);
            [s1_8k,lev1]=activlev(s1_8k,fs8k,'n'); % y_norm = y /sqrt(lev);
            s2_8k=resample(s2,fs8k,fs);
            [s2_8k,lev2]=activlev(s2_8k,fs8k,'n');
                        
            weight_1=10^(inwav1_snr/20);
            weight_2=10^(inwav2_snr/20);
            
            s1_8k = weight_1 * s1_8k;
            s2_8k = weight_2 * s2_8k;
            
            switch min_max{i_mm}
                case 'max'
                    mix_8k_length = max(length(s1_8k),length(s2_8k));
                    s1_8k = cat(1,s1_8k,zeros(mix_8k_length - length(s1_8k),6));
                    s2_8k = cat(1,s2_8k,zeros(mix_8k_length - length(s2_8k),6));
                case 'min'
                    mix_8k_length = min(length(s1_8k),length(s2_8k));
                    s1_8k = s1_8k(1:mix_8k_length,:);
                    s2_8k = s2_8k(1:mix_8k_length,:);
            end
            mix_8k = s1_8k + s2_8k;
                    
            max_amp_8k = max(cat(1,abs(mix_8k(:)),abs(s1_8k(:)),abs(s2_8k(:))));
            mix_scaling_8k = 1/max_amp_8k*0.9;
            s1_8k = mix_scaling_8k * s1_8k;
            s2_8k = mix_scaling_8k * s2_8k;
            mix_8k = mix_scaling_8k * mix_8k;
            
            % apply same gain to 16 kHz file
            s1_16k = weight_1 * s1 / sqrt(lev1);
            s2_16k = weight_2 * s2 / sqrt(lev2);
            
            switch min_max{i_mm}
                case 'max'
                    mix_16k_length = max(length(s1_16k),length(s2_16k));
                    s1_16k = cat(1,s1_16k,zeros(mix_16k_length - length(s1_16k),6));
                    s2_16k = cat(1,s2_16k,zeros(mix_16k_length - length(s2_16k),6));
                case 'min'
                    mix_16k_length = min(length(s1_16k),length(s2_16k));
                    s1_16k = s1_16k(1:mix_16k_length,:);
                    s2_16k = s2_16k(1:mix_16k_length,:);
            end
            mix_16k = s1_16k + s2_16k;
            
            max_amp_16k = max(cat(1,abs(mix_16k(:)),abs(s1_16k(:)),abs(s2_16k(:))));
            mix_scaling_16k = 1/max_amp_16k*0.9;
            s1_16k = mix_scaling_16k * s1_16k;
            s2_16k = mix_scaling_16k * s2_16k;
            mix_16k = mix_scaling_16k * mix_16k;            
            
            % save 8 kHz and 16 kHz mixtures, as well as
            % necessary scaling factors
            
            scaling_16k(i,1) = weight_1 * mix_scaling_16k/ sqrt(lev1);
            scaling_16k(i,2) = weight_2 * mix_scaling_16k/ sqrt(lev2);
            scaling_8k(i,1) = weight_1 * mix_scaling_8k/ sqrt(lev1);
            scaling_8k(i,2) = weight_2 * mix_scaling_8k/ sqrt(lev2);
            
            scaling16bit_16k(i) = mix_scaling_16k;
            scaling16bit_8k(i)  = mix_scaling_8k;
                        
            audiowrite([output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name sprintf('_roomGeometry_%03d', n) '.wav'],s1_8k,fs8k);
            audiowrite([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name sprintf('_roomGeometry_%03d', n) '.wav'],s1_16k,fs);
            audiowrite([output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name sprintf('_roomGeometry_%03d', n) '.wav'],s2_8k,fs8k);
            audiowrite([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name sprintf('_roomGeometry_%03d', n) '.wav'],s2_16k,fs);
            audiowrite([output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name sprintf('_roomGeometry_%03d', n) '.wav'],mix_8k,fs8k);
            audiowrite([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name sprintf('_roomGeometry_%03d', n) '.wav'],mix_16k,fs);
            
            if mod(i,10)==0
                fprintf(1,'.');
                if mod(i,200)==0
                    fprintf(1,'\n');
                end
            end
            
        end
        save([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_8k','scaling16bit_8k');
        save([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_16k','scaling16bit_16k');
        
        fclose(fid);
        fclose(fid_s1);
        fclose(fid_s2);
        fclose(fid_m);
    end
end
