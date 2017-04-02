function layer=tensor_init_gpu(input_shape,type,loss)
%% A tensor layer ,can be a input layer or a ouyput alyer
%% Basic layer attributes
%Input tensor sahpe
layer.input_shape=input_shape;
%Output tensor shape
layer.output_shape=input_shape;
%The type of the layer
layer.type=type;
layer.output=zeros(input_shape,'single','gpuArray');
if nargin<=2
    return
elseif nargin>2
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
end