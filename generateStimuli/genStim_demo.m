clear all
clc

par.Fs = 44100;           % Sampling frequency     
par.duration = 20;        % the length for building signal, unit in second
par.stimDuration = 4;     % the length of desired stimulus length, unit in second

% par.s = 0.35; % width of the lognormal distribution, specified below
% par.m = 0.6; % mode of the lognormal distribution, specified below

par.slope = 1; % slope of the 1/f noise of the AM spectrum
par.amAddNoise = 'multiply'; % 'add' or 'multiply'

par.doFilter=0; % stim_20200718_noFilter
par.doDetrend=0;

par.rampLength = 0.1; % unit second

par.filterFile = 'parameters_20210216/lowpass15Hz_coef_20200603';

numStimGen = 1; % number of stimuli generated per condition

%% generate stimuli

for m = [0.6, 2.4, 4.2]
    par.m = m; % % mode of the lognormal distribution
    for s = [0.15,0.35,0.55]
        par.s = s;
        for n = 1:numStimGen
            par.randSeed = randi(1e5); % random seed, integer between 1 and 10000
            saveName = ['stimuli_demo/lnn_AM',num2str(par.m),'Hz_s',num2str(par.s*100),'_',num2str(n),'.wav'];
            disp(saveName);
            genLogNormStim_ShiftAmp_20s_demo(par,saveName);
        end
    end
end


%% equalize the loudness with RMS

files = dir('stimuli_demo/lnn_AM*.wav');
for k = 1:length(files)
	stim_all(k,:) = audioread(['stimuli_demo/',files(k).name]);
end

stim_rms = rms(stim_all,2); % obtain the RMS
stim_equal = stim_all./stim_rms * min(stim_rms); % equalize RMS to the min
stim_equal_rms = rms(stim_equal,2); % check whether the RMSs are the same

if sum(sum(abs(stim_equal)>1)) % check whether any abs amplitude > 1
    disp('amplitude > 1, fix it!')
end

% save the loudness-equalized audio files
for k = 1:length(files)
	audiowrite(['stimuli_demo/equal_',files(k).name(1:end-4),'.m4a'],stim_equal(k,:),par.Fs);
end




%% plot logNormalModel

load('stimuli_demo/lnn_AM0.6Hz_s15_4_fig.mat')

figure;
subplot(2,1,1);
plot(fig.logNormModel.x,fig.logNormModel.y,'linewidth',2);
xlim([0.25,32])
xlabel('Hz')
ylabel('normalized amplitude')
legend(['mode: ',num2str(fig.logNormModel.m),' Hz'])
set(gca,'FontSize',14)
grid on

subplot(2,1,2);
semilogx(fig.logNormModel.x,fig.logNormModel.y,'linewidth',2);
xlim([0.25,32])
xlabel('Hz')
ylabel('normalized amplitude')
legend(['mode: ',num2str(fig.logNormModel.m),' Hz'])
h = gca;
h.XTick = [0.5:0.5:4,5,6:2:10,12:4:32];
set(gca,'FontSize',14)
grid on

%% plot add 1/f noise

load('stimuli_demo/lnn_AM0.6Hz_s15_4_fig.mat')


figure;
subplot(2,1,1);
plot(fig.addSlope.x,fig.addSlope.y11,'linewidth',2);
hold on
plot(fig.addSlope.x,fig.addSlope.pNoise,'linewidth',2);
xlim([0.25,32])
xlabel('Hz')
ylabel('normalized amplitude')
legend(['mode: ',num2str(fig.addSlope.m),' Hz'],['1/f^{',num2str(fig.addSlope.slope) ,'} noise'])
title(['AM envelope spectrum (linear), ',fig.addSlope.amAddNoise,' 1/f^{',num2str(fig.addSlope.slope) ,'} noise'])
set(gca,'FontSize',14)
grid on


subplot(2,1,2);
semilogx(fig.addSlope.x,20*log10(fig.addSlope.y11),'linewidth',2);
hold on
plot(fig.addSlope.x,20*log10(fig.addSlope.pNoise),'linewidth',2);
h = gca;
h.XTick = [0.5:0.5:4,5,6:2:10,12:4:32];
xlim([0.25,32])
xlabel('Hz')
ylabel('dB')
legend(['mode: ',num2str(fig.addSlope.m),' Hz'],['1/f^{',num2str(fig.addSlope.slope) ,'} noise'],'location','southwest')
title(['AM envelope spectrum (dB), ',fig.addSlope.amAddNoise,' 1/f^{',num2str(fig.addSlope.slope) ,'} noise'])
set(gca,'FontSize',14)
grid on


%% plot example stimuli




col = lines(4);

load('stimuli_demo/lnn_AM2.4Hz_s35_1_fig.mat')
figure('position',[100,100,1000,600])
semilogx(fig.logNormModel.x,fig.logNormModel.y,'linewidth',4,'color',col(4,:));
xlim([0.1,24])
xlabel('frequency (Hz)')
ylabel('normalized power')
title('Lognormal (peak AM frequency, \sigma) \rightarrow AM spectrum')
legend('AM spectrum','fontsize',20,'box','off')
set(gca,'XTick',[0.1,0.5,1,5,10,20],'XMinorTick','off','YTick',[0,0.5,1],'FontSize',14)

% grid on


figure('position',[100,100,1200,600])

subplot(2,2,4)
load('stimuli_demo/lnn_AM0.6Hz_s15_1_fig.mat')
plot(fig.sound.tStim,fig.sound.am1,'color',[0.5,0.5,0.5]);
hold on
plot(fig.sound.tStim(4410:end-4410),fig.envFilt.env_proc(4410:end-4410)*1.2+0.05,'color',col(4,:),'linewidth',3);
title(['peak AM frequency: ',num2str(fig.sound.m),' Hz, \sigma: ',num2str(fig.logNormModel.s)])
set(gca,'FontSize',14)
ylim([-1,1]*1.1)
xlabel('time (s)')


subplot(2,2,3)
load('stimuli_demo/lnn_AM0.6Hz_s55_1_fig.mat')
plot(fig.sound.tStim,fig.sound.am1,'color',[0.5,0.5,0.5]);
hold on
plot(fig.sound.tStim(4410:end-4410),fig.envFilt.env_proc(4410:end-4410)*1.2+0.05,'color',col(4,:),'linewidth',3);
title(['peak AM frequency: ',num2str(fig.sound.m),' Hz, \sigma: ',num2str(fig.logNormModel.s)])
set(gca,'FontSize',14)
ylim([-1,1]*1.1)
xlabel('time (s)')


subplot(2,2,2)
load('stimuli_demo/lnn_AM4.2Hz_s15_1_fig.mat')
plot(fig.sound.tStim,fig.sound.am1,'color',[0.5,0.5,0.5]);
hold on
plot(fig.sound.tStim(4410:end-4410),fig.envFilt.env_proc(4410:end-4410)*1.2+0.05,'color',col(4,:),'linewidth',3);
title(['peak AM frequency: ',num2str(fig.sound.m),' Hz, \sigma: ',num2str(fig.logNormModel.s)])
set(gca,'FontSize',14)
ylim([-1,1]*1.1)
% xlabel('time (s)')


subplot(2,2,1)
load('stimuli_demo/lnn_AM4.2Hz_s55_1_fig.mat')
plot(fig.sound.tStim,fig.sound.am1,'color',[0.5,0.5,0.5]);
hold on
plot(fig.sound.tStim(4410:end-4410),fig.envFilt.env_proc(4410:end-4410)*1.2+0.05,'color',col(4,:),'linewidth',3);
title(['peak AM frequency: ',num2str(fig.sound.m),' Hz, \sigma: ',num2str(fig.logNormModel.s)])
set(gca,'FontSize',14)
ylim([-1,1]*1.1)
% xlabel('time (s)')


