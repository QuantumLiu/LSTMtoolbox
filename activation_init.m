function layer= activation_init( prelayer,act_fun ,loss)
%% Basic layer attributes
layer.type='activation';

layer.prelayer_type=prelayer.type;

layer.batch=1;
layer.epoch=1;

layer.input_shape=prelayer.output_shape;
layer.output_shape=prelayer.output_shape;

layer.input=prelayer.output;
layer.output=prelayer.output;

if ~strcmpi(layer.prelayer_type,'input')
    layer.dx=layer.input;
end
layer.e=layer.output;

if nargin>2
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
layer.act=@(x)act(x,act_fun); 
layer.dact=@(x)dact(x,act_fun); 
end

