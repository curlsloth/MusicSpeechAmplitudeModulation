function [fig] = genLogNormStim_ShiftAmp_20s_demo(par,saveName)
%% 2021/03/20 revised
% add x-scale shifting parameter b

%% parameters

Fs = par.Fs;           % Sampling frequency     
duration = par.duration;        % the length for building signal, unit in second
stimDuration = par.stimDuration;     % the length of desired stimulus length, unit in second

s = par.s; % width of the lognormal distribution
m = par.m; % mode of the lognormal distribution

slope = par.slope; % slope of the 1/f noise of the AM spectrum
amAddNoise = par.amAddNoise; % 'add' or 'multiply'

doFilter = par.doFilter;
doDetrend = par.doDetrend;

rampLength = par.rampLength; % unit second

rng(par.randSeed)

load(par.filterFile)
load('parameters_20210216/lnn_20s_20210216')
% load('parameters_20210216/lnn_pinkFiltered_20s_20210325')
load('parameters_20210216/interpCoef20210318')

%% generate lognormal distribution

% mode: exp(mu-sigma^2) https://en.wikipedia.org/wiki/Log-normal_distribution
% mode = 4: exp(1.3863)
% mode = 1.5: exp(0.4055)


T = 1/Fs;                               % Sampling period       
Lsignal = duration*Fs;                  % Length of signal
Lstim = stimDuration*Fs;                % Length of sound
tSignal = (0:Lsignal-1)*T;              % Time vector
tStim = (0:Lstim-1)*T;                  % Time vector
fSignal = Fs*(0:(Lsignal/2))/Lsignal;   % frequency vector of signal
fStim = Fs*(0:(Lstim/2))/Lstim;         % frequency vector of sound

% translate mode to mu, a parameter of lognormal function
% considering the x-scale shifting parameter b, see file '/Users/andrewchang/PoeppelStudies/modelMusicSpeechCorpus/compareFitresult.m'
b = m*interpCoef(1)+interpCoef(2);
mu = log(m-b)+s^2;

x = (fSignal)';
y = lognpdf(x-b,mu,s);
y = y./max(y);

% save variables for plot
fig.logNormModel.x = x;
fig.logNormModel.y = y;
fig.logNormModel.m = m;
fig.logNormModel.s = s;

%% add 1/f noise

pNoise = sqrt((1./fSignal).^slope');

if strcmp(amAddNoise,'add')
    % using addition to add the 1/f noise (assuming that it is independent of the signal)
    y11 = amSNR.*y + pNoise;
    
elseif strcmp(amAddNoise,'multiply')
    % using mutiplication to add 1/f noise (assuming that it is modulating the signal)
    y11 = y.*pNoise;
end

y11 = y11./max(y11(2:end));



% save variables for plot
fig.addSlope.x = x;
fig.addSlope.m = m;
fig.addSlope.y11 = y11;
fig.addSlope.pNoise = pNoise./pNoise(2);
fig.addSlope.amAddNoise = amAddNoise;
fig.addSlope.slope = slope;




%% iFFT


rng(par.randSeed); % set up the random seed
phase1=rand(1,(duration*Fs)/2-1)*2*pi;
tempSig1 = y11(2:(duration*Fs)/2)'.*(cos(phase1)+1i*sin(phase1)); % skip the first position of y11, which is 0 Hz
if mod(Lsignal,2)==0 % if the length of the tempSig is even
    s1 = [0, tempSig1, 0, fliplr(conj(tempSig1)) ]; % DC, spectrum, Nyquest frequency (a real number), right-left flipped spectrum 
elseif mod(Lsignal,2)==1 % if the length of the tempSig is odd
    s1 = [0, tempSig1, fliplr(conj(tempSig1)) ]; % DC, spectrum, right-left flipped spectrum 
end
% inverse fourier transform
env_orig = ifft(s1); 
env_orig = env_orig-min(env_orig); % make the envelope values positive
env_orig = env_orig./max(env_orig); % make the max value 1
env_orig = sqrt(env_orig); %%%%%% transform power to amplitude %%%%%%


%% lowpass filter the envelope

if doFilter == 1
    env_proc = filtfilt(b, 1, env_orig);
else
    env_proc = env_orig;
end


if doDetrend==1
    % detrend
    env_proc = detrend(env_proc);
end

env_20s = env_proc; % save a copy of 20 sec signal

Lstim = stimDuration*Fs;
Lsignal = duration*Fs;
if Lstim < Lsignal % took only the middle part of the vector
	lenDiff = Lsignal-Lstim;
	env_proc = env_proc(round(lenDiff/2)+1:round(lenDiff/2)+Lstim);
    env_orig = env_orig(round(lenDiff/2)+1:round(lenDiff/2)+Lstim);
    carrierLNN_stim = carrierLNN(round(lenDiff/2)+1:round(lenDiff/2)+Lstim);
end

% normalization
env_proc = env_proc-min(env_proc);
env_proc = env_proc./max(env_proc) * 0.8;

env_orig = env_orig-min(env_orig);
env_orig = env_orig./max(env_orig) * 0.8;

env_20s = env_20s-min(env_20s);
env_20s = env_20s./max(env_20s) * 0.8;

env_orig = env_orig';
env_proc = env_proc';
env_20s = env_20s';


% save variables for plot
fig.envFilt.doFilter = doFilter;
fig.envFilt.tStim = tStim;
fig.envFilt.tSignal = tSignal;
fig.envFilt.env_orig = env_orig;
fig.envFilt.env_proc = env_proc;
fig.envFilt.env_20s = env_20s;
fig.envFilt.m = m;


%% build stimulus


cycleLength = rampLength*4; % length of the ramp equals to 1/4 of the cycle
fRamp = 1/cycleLength; % frequency of the ramp

ramp_up = sin(2*pi*fRamp*(1/Fs:1/Fs:rampLength))';
ramp_down = cos(2*pi*fRamp*(1/Fs:1/Fs:rampLength))';

env_proc(1:length(ramp_up)) = env_proc(1:length(ramp_up)).*ramp_up;
env_proc(end-length(ramp_down)+1:end) =env_proc(end-length(ramp_down)+1:end).*ramp_down;


% adjust the loudness later
am = carrierLNN_stim .* env_proc;
am = am./max(abs(am));

am_20s = carrierLNN .* env_20s;
am_20s = am_20s./max(abs(am_20s));


% save variables for plot
fig.sound.tStim = tStim;
fig.sound.am1 = am;
fig.sound.m = m;

fig.sound.tSignal= tSignal;
fig.sound.am_20s = am_20s;

audiowrite(saveName,am,Fs);

% % I found it's better to equalize the RMS after all the sounds are generated.
% am2 = am1/rms(am1)*0.25; % make the root-mean-square of the sound to be 0.25
% audiowrite(saveName,am2,Fs);

%% sanity check

am=am-mean(am);
envCheck_4s=abs(hilbert(am));


am_20s=am_20s-mean(am_20s);
envCheck_20s=abs(hilbert(am_20s));


% compare the AM envelope spectrums: model vs. stimulus
specEnv_4s = fft(envCheck_4s.^2); % take square to transform amp to power
P2_env_4s = abs(specEnv_4s/Lstim);
P1_env_4s = P2_env_4s(1:Lstim/2+1);
P1_env_4s(2:end-1) = 2*P1_env_4s(2:end-1);
P1_env_4s = P1_env_4s./max(P1_env_4s(2:end-1));


specEnv_20s = fft(envCheck_20s.^2); % take square to transform amp to power
P2_env_20s = abs(specEnv_20s/Lsignal);
P1_env_20s = P2_env_20s(1:Lsignal/2+1);
P1_env_20s(2:end-1) = 2*P1_env_20s(2:end-1);
P1_env_20s = P1_env_20s./max(P1_env_20s(2:end-1));



% save variables for plot
fig.check.tStim = tStim;
fig.check.envCheck_4s = envCheck_4s;
fig.check.env_proc = env_proc;
fig.check.m = m;
fig.check.fStim = fStim;
fig.check.fSignal = fSignal;
fig.check.y11 = y11;
fig.check.P1_env_4s = P1_env_4s;

fig.check.tSignal = tSignal;
fig.check.env_20s = env_20s;
fig.check.P1_env_20s = P1_env_20s;

figName = [saveName(1:end-4),'_fig.mat'];

save(figName,'fig','par')

end

