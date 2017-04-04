function test_lstm(nb_batch,hiddensize,input_dim,timestep,batch_size,nb_epoch)
lstm_ff=@(layer,prelayer)lstm_ff_gpu(layer,prelayer);
lstm_bp=@(layer,next_layer)lstm_bp_gpu(layer,next_layer);
act_ff=@(layer,prelayer)activation_ff(layer,prelayer);
act_bp=@(layer,next_layer)activation_bp(layer,next_layer);
pars.learningrate=0.01;
pars.momentum=0;
pars.opt='sgd';
optimize=@(layer)layer_optimize(layer,pars);
x=ones(input_dim,timestep,batch_size*nb_batch,'single','gpuArray');
y=(zeros(hiddensize(end),timestep,batch_size*nb_batch,'single','gpuArray'));
y(1,:,:)=1;
inputlayer=tensor_init_gpu([input_dim,timestep,batch_size],'input');
lstmlayer1=lstm_init_gpu(inputlayer,hiddensize(1),1);
lstmlayer2=lstm_init_gpu(lstmlayer1,hiddensize(2),1);
lstmlayer3=lstm_init_gpu(lstmlayer2,hiddensize(3),1);
outputlayer=activation_init(lstmlayer3,'softmax','categorical_cross_entropy');
profile on;
for epoch=1:nb_epoch
    tic;
    for i=1:nb_batch
        lstmlayer1=lstm_ff(lstmlayer1,x(:,:,(i-1)*batch_size+1:i*batch_size));
        lstmlayer2=lstm_ff(lstmlayer2,lstmlayer1);
        lstmlayer3=lstm_ff(lstmlayer3,lstmlayer2);
        outputlayer=act_bp(eval_loss(act_ff(outputlayer,lstmlayer3),y(:,:,(i-1)*batch_size+1:i*batch_size)),[]);
        lstmlayer3=optimize(lstm_bp(lstmlayer3,outputlayer));
        lstmlayer2=optimize(lstm_bp(lstmlayer2,lstmlayer3));
        lstmlayer1=optimize(lstm_bp(lstmlayer1,lstmlayer2));
    end
    toc;
end
profile report;
plot(outputlayer.loss);
end