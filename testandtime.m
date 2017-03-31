function testandtime(nb_batch,hiddensize,input_dim,timestep,batch_size)
x=rand(input_dim,timestep,batch_size*nb_batch,'single','gpuArray');
y=ones(hiddensize,timestep,batch_size*nb_batch,'single','gpuArray');
opts.learningrate=0.1;
opts.momentum=0;
opts.conected_type='input';
loss=zeros(nb_batch,1,'single','gpuArray');
lstmlayer=lstm_init_gpu(input_dim,timestep,batch_size,hiddensize,opts);
profile on;
tic;
for i=1:nb_batch
lstmlayer=lstm_ff_gpu(lstmlayer,x(:,:,(i-1)*batch_size+1:i*batch_size));
e=y(:,:,(i-1)*batch_size+1:i*batch_size).^2-lstmlayer.output.^2;
loss(i)=mean(mean(mean(e)));
lstmlayer=lstm_bp_gpu(lstmlayer,e);
lstmlayer=lstm_update(lstmlayer,i);
disp(i);
end;toc;profile report;
plot(loss,'DisplayName','loss')