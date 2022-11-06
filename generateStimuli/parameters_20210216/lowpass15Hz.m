% % filter of the envelope
% Equiripple Lowpass filter designed using the FIRPM function.

% All frequency values are in Hz.

Fpass = 15;              % Passband Frequency
Fstop = 25;              % Stopband Frequency
Dpass = 0.05;            % Passband Ripple
Dstop = 0.0001;          % Stopband Attenuation
dens  = 20;              % Density Factor
Fs = 44100;

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd = dfilt.dffir(b);

% % plot the filter
% h = fvtool(Hd,'Analysis','magnitude');figure(gcf);h.Fs = Fs;

save('lowpass15Hz_coef_20200603','b')