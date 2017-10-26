[data, sample_rate] = audioread('dataSamples/audio/audio_audio.wav');
range = max(data) - min(data);
data = (data - min(data))./range;
data = data .* 2 - 1;
mex 'external_libs/rpde/close_ret.c';
[H_Norm, rpd] = rpde(data,4,35,0.12);

mex 'external_libs/fastdfa/fastdfa_core.c';
[alpha, intervals, flucts] = fastdfa(data);

