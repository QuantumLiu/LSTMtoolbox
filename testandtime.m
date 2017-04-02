function testandtime(nb_batch,hiddensize,input_dim,timestep,batch_size,nb_epoch)
tensorlayer_ff=@(layer,input)tensor_ff_gpu(layer,input);
lstm_ff=@(layer,prelayer)lstm_ff_gpu(layer,prelayer);
lstm_bp=@(layer,next_layer)lstm_bp_gpu(layer,next_layer);
act_ff=@(layer,prelayer)activation_ff(layer,prelayer);
act_bp=@(layer,next_layer)activation_bp(layer,next_layer);
pars.learningrate=0.01;
pars.momentum=0;
pars.opt='mse';
optimize=@(layer)layer_optimize(layer,pars);
x=rand(input_dim,timestep,batch_size*nb_batch,'single','gpuArray');
y=(zeros(hiddensize,timestep,batch_size*nb_batch,'single','gpuArray'));
y(1,:,:)=1;
inputlayer=tensor_init_gpu([input_dim,timestep,batch_size],'input');
lstmlayer=lstm_init_gpu(inputlayer,hiddensize,'mse');
outputlayer=activation_init(lstmlayer,'softmax','categorical_cross_entropy');
profile on;
for epoch=1:nb_epoch
    tic;
    for i=1:nb_batch
        lstmlayer=lstm_ff(lstmlayer,tensorlayer_ff(inputlayer,x(:,:,(i-1)*batch_size+1:i*batch_size)));
        lstmlayer=optimize(lstm_bp(lstmlayer,act_bp(eval_loss(act_ff(outputlayer,lstmlayer),y(:,:,(i-1)*batch_size+1:i*batch_size)),[])));
    end
    toc;
end
profile report;
plot(lstmlayer.loss,'DisplayName','loss')