regaud = audioread('test_matlab30.wav');
beam = audioread('enhanced90.wav');
fs = 44100;
regaud = regaud(:,1);
figure
plot(regaud)
title('regular')

figure
plot(beam)
title('beam')


    
m1 = length(regaud);
m2 = length(beam);

n1 = pow2(nextpow2(m1));
n2 = pow2(nextpow2(m2));
y1 = fft(regaud,n1);
y2 = fft(beam,n2);

f1 = (0:n1-1)*(fs/n1)/10;
f2 = (0:n2-1)*(fs/n2)/10;

power1= abs(y1).^2/n1;
power2 = abs(y2).^2/n2;

figure
hpsd1 = dspdata.psd(power1(1:length(power1)/2),'Fs',fs);
plot(hpsd1)
title('original')

figure
hpsd2 = dspdata.psd(power2(1:length(power2)/2),'Fs',fs);
plot(hpsd2)
title('beamformed')


figure
spectrogram(regaud)
title('original')

figure
spectrogram(beam)
title('beamformed')

%using  pwelch
figure
pwelch(regaud);

figure
pwelch(beam);




% figure
% plot(f1(1:floor(n1/2)),power1(1:floor(n1/2)))
% xlabel('Frequency')
% ylabel('Power')
% 
% 
% figure
% plot(f2(1:floor(n2/2)),power2(1:floor(n2/2)))
% xlabel('Frequency')
% ylabel('Power')



