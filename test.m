opts.conected_type='input';
input=rand(100,10,32,'single','gpuArray');
opts.learningrate=0.1;
opts.momentum=0;
lstmlayer=lstm_init_gpu(100,10,32,512,opts);
profile on;
tic;
for i=1:100
lstmlayer=lstm_ff_gpu(lstmlayer,input);
e=1-lstmlayer.xh(102:end,2:end,:).^2;
loss(i)=mean(mean(mean(sqrt(1-lstmlayer.xh(102:end,2:end,:).^2))));
lstmlayer=lstm_bp_gpu(lstmlayer,e);
lstmlayer=lstm_update(lstmlayer,i);disp(i);
end;toc;profile report;
plot(loss,'DisplayName','loss')