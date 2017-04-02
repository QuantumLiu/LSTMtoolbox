function testandtime(nb_batch,hiddensize,input_dim,timestep,batch_size,nb_epoch)
tensorlayer_ff=@(layer,input)tensor_ff_gpu(layer,input);
lstm_ff=@(layer,prelayer)lstm_ff_gpu(layer,prelayer);
lstm_bp=@(layer,next_layer)lstm_bp_gpu(layer,next_layer);
pars.learningrate=0.01;
pars.momentum=0;
pars.opt='mse';
update=@(layer)layer_update(layer,pars);
x=ones(input_dim,timestep,batch_size*nb_batch,'single','gpuArray');
y=5*sin(ones(hiddensize,timestep,batch_size*nb_batch,'single','gpuArray'));
inputlayer=tensor_init_gpu([input_dim,timestep,batch_size],'input');
%outputlayer=tensor_init_gpu([hiddensize,timestep,batch_size],'output','mse');
lstmlayer=lstm_init_gpu(inputlayer,hiddensize,'mse');
profile on;
for epoch=1:nb_epoch
    tic;
    for i=1:nb_batch
        lstmlayer=update(lstm_bp(eval_loss(lstm_ff(lstmlayer,tensorlayer_ff(inputlayer,x(:,:,(i-1)*batch_size+1:i*batch_size))),y(:,:,(i-1)*batch_size+1:i*batch_size)),[]));
    end
    toc;
end
profile report;
plot(lstmlayer.loss,'DisplayName','loss')