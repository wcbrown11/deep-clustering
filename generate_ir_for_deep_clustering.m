procFs = 16000;                                                               % Sampling frequency (Hz)
c = 343;                                                                      % Sound velocity (m/s)
nsample = 4096;                                                                % Length of desired RIR
N_harm = 40;                                                                  % Maximum order of harmonics to use in SHD
K = 2;                                                                        % Oversampling factor

for n = 1:1
   L = [4 4 3] + rand(1,3).*[4 4 2];                                          % Room dimensions (x,y,z) in m
   sphLocation = [L(1:2)/2 2] + 1/4*randn(1,3).*[2 2 1];                      % Receiver location (x,y,z) in m
   sphRadius = 0.02 + rand(1)*0.08;                                           % Radius of the sphere (m)
   s = sphLocation + 1/4*randn(2,3).*[1 1 .5];                                % Source location(s) (x,y,z) in m
   while norm(sphLocation - s(1,:)) <= sphRadius || norm(sphLocation - s(2,:)) <= sphRadius
      s = sphLocation + 1/4*randn(2,3).*[1 1 .5];                                % Source location(s) (x,y,z) in m
   end
   beta = [0.2 0.2 0.2 0.2 0 0] + rand(1,6).*[0.75 0.75 0.75 0.75 0.5 0.5];   % Room reflection coefficients [\beta_x_1
                                                                              % \beta_x_2 \beta_y_1 \beta_y_2 \beta_z_1 \beta_z_2]
   order = -1;                                                                % Reflection order (-1 is maximum reflection order)
   
   sphType = 'open';                                                          % Type of sphere (open/rigid)
   result = zeros(12,4096);
   mic = [0 pi/2; pi/3 pi/2; pi*2/3 pi/2; pi pi/2; pi*4/3 pi/2; pi*5/3 pi/2] + [2*pi*rand(1) pi/2]; % Microphone positions (azimuth, elevation)
   for p = 1:6
      [result(p,:), H1, beta_hat1] = smir_generator(c, procFs, sphLocation, s(1,:), L, beta, sphType, sphRadius, mic(p,:), N_harm, nsample, K, order);
      [result(p+6,:), H2, beta_hat2] = smir_generator(c, procFs, sphLocation, s(2,:), L, beta, sphType, sphRadius, mic(p,:), N_harm, nsample, K, order);
   end
   % Save all variables into a .mat file named sprintf('roomGeometry_%03d', n)
   mat_name = char(sprintf('/scratch/near/roomGeometry/roomGeometry_%03d.mat', n));
   save(mat_name);
   
   %result(1:6,:) = h1;
   %result(7:12,:) = h2;
   % Write two multichannel wav files, one for source 1, one for source 2
   audiowrite(sprintf('/scratch/near/roomGeometry/roomGeometry_%03d.wav', n),0.1*result.',procFs);
end