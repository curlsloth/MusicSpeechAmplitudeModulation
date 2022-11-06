% generate 20 s lownoise noise
% Because the LNN will repeating itself after 1 s, I generate 20 1-s LNN and then concatenate them.

fs = 44100;
dur = 1; % duration for each excerpt
nExcerpt = 20; % number of excerpts
tempArray = zeros(nExcerpt,dur*fs);
for n = 1:nExcerpt % total 4 seconds
    tempArray(n,:)= GenLowNoise2(dur, 20, 20000, fs);
end


scaleFactor = 0.25./rms(tempArray,2); % get the amplitude scale of each noise, relative to 0.25
tempArray = tempArray.*scaleFactor; % normalize amplitude of each noise to 0.25
tempArray = tempArray';

carrierLNN = tempArray(:); % concatinate all excerpts

save('lnn_20s_20210216.mat','carrierLNN')


%% plot the results for checking
% y = fft(carrierNoise3);
% 
% P2 = abs(y/length(y));
% P1 = P2(1:length(y)/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% P1 = P1./max(P1(2:end-1));
% f = fs*(0:(length(y)/2))/length(y);   % frequency vector
% 
% semilogx(f,P1)
% loglog(f,P1)