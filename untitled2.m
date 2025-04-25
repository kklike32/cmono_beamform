regaud = audioread('test_matlab30.wav');
beam1 = audioread('enhanced30.wav');
beam2 = audioread('enhanced50.wav');
beam3 = audioread('enhanced80.wav')
fs = 44100;
regaud = regaud(:,1);
% figure
% plot(regaud)
% title('regular')
% % 
% figure
% plot(beam)
% title('beam2')
% 
figure
pwelch(regaud)

figure
pwelch(beam1)
% 
% 
% figure
% pwelch(beam2)
% 
% figure
% pwelch(beam3)

% figure
% polarplot(regaud)