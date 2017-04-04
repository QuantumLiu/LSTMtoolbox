function layer=dense_init_gpu(prelayer,hiddensize,loss)
%% Basic layer attributes
%Input tensor sahpe
layer.input_shape=prelayer.output_shape;
dim=prelayer.output_shape(1);
batchsize=prelayer.output_shape(2);
layer.type='dense';
layer.prelayer_type=prelayer.type;
layer.output_shape=[hiddensize,batchsize];
layer.hiddensize=hiddensize;
layer.batchsize=batchsize;
layer.batch=1;
layer.epoch=1;
%% Dense layer attributes
%W contains weights bias
layer.weights_dim=dim+1;
layer.W=rand([hiddensize,layer.weights_dim],'single','gpuArray')-0.5;
layer.input=ones([layer.input_shape(1)+1,batchsize],'single','gpuArray');
layer.output=zeros(layer.output_shape,'single','gpuArray');
if ~strcmpi(layer.prelayer_type,'input')
    layer.dx=zeros(layer.input_shape,'single','gpuArray');
end
layer.e=layer.output;
if nargin<=2
    return
elseif nargin>2
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
end

