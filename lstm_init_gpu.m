function layer=lstm_init_gpu(prelayer,hiddensize,return_sequence,loss)
%% Basic layer attributes
%Input tensor sahpe
layer.input_shape=prelayer.output_shape;
dim=prelayer.output_shape(1);
timestep=prelayer.output_shape(2);
batchsize=prelayer.output_shape(3);
if nargin<3
    return_sequence=1;
end
layer.return_sequence=return_sequence;
if return_sequence
%Output tensor shape
    layer.output_shape=[hiddensize,timestep,batchsize];
else
    layer.output_shape=[hiddensize,batchsize];        
end
%The type of the layer
layer.type='lstm';
%conected layer type
layer.prelayer_type=prelayer.type;
%The hiddensize of the layer
layer.hiddensize=hiddensize;

layer.batch=1;
layer.epoch=1;
%% lstm layer attributes
%Timestep 
layer.timestep=timestep;
layer.batchsize=batchsize;
%n is the number of unrolled timesteps in one batch
layer.n=batchsize*timestep;
%Put x(t) and h(t) in one array 
layer.xh=ones([dim+1+hiddensize,timestep+1,batchsize],'single','gpuArray');
%W is the weights of all four gates and bias
layer.weights_dim=dim+1+hiddensize;
layer.W=rand([4*hiddensize,layer.weights_dim],'single','gpuArray')-0.5;
%Compute the value of x_t*wx_t for all ts in one time
layer.maX=zeros([4*hiddensize,timestep,batchsize],'single','gpuArray');
%value before activited
layer.ma=layer.maX;
%value activited
layer.mb=layer.maX;
%sc:state of cell
layer.sc=zeros([hiddensize,timestep,batchsize],'single','gpuArray');
layer.bc=layer.sc;
%The output tensor and error
layer.output=zeros(layer.output_shape,'single','gpuArray');
layer.e=layer.sc;
%diffs
layer.dW=zeros(size(layer.W),'single','gpuArray');
layer.dma=zeros([4*hiddensize,timestep+1,batchsize],'single','gpuArray');
layer.dmb=layer.dma;
layer.dsc=layer.sc;
layer.dh=layer.dsc;
if ~strcmpi(layer.prelayer_type,'input')
    layer.dx=zeros(layer.input_shape,'single','gpuArray');
end
if nargin>3
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
%% methods
layer.act_f =@(x)act(x,'sigmoid'); % active function for gate
layer.act_tc =@(x)act(x, 'tanh'); % active function for tc
layer.act_h = @(x)act(x, 'tanh');
layer.dact_f= @(x)dact(x,'sigmoid');
layer.dact_tc =@(x)dact(x, 'tanh'); % active function for tc
layer.dact_h = @(x)dact(x, 'tanh');
layer.ff=@(layer,prelayer)lstm_ff_gpu(layer,prelayer);
layer.bp=@(layer,next_layer)lstm_bp_gpu(layer,next_layer);
layer.trainable=1;
layer.opt=@(layer,pars)layer_optimize(layer,pars);
end