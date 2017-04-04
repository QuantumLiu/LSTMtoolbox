function model=model_init(input_shape,configs,optimizer)
model.layers=cell(1,length(configs)+1);
model.layers{1}=tensor_init_gpu(input_shape,'input');
for l=2:length(model.layers)
    model.layers{l}=layer_init(model.layers{l-1},configs{l-1});
end
model.layers=[model.layers,0];

model.input_shape=model.layers{1}.input_shape;
model.output_shape=model.layers{end-1}.output_shape;
model.batchsize=model.input_shape(end);
model.loss=[];
model.optimize=@(layer,batch,epoch)layer_optimize(layer,optimizer,batch,epoch);
model.eval_loss=@(outputlayer,y_true)eval_loss(outputlayer,y_true);
end
function layer=layer_init(prelayer,config)
switch config.type
    case 'lstm'
        if isfield(config,'loss')
        layer=lstm_init_gpu(prelayer,config.hiddensize,config.return_sequence,config.loss);
        else
            layer=lstm_init_gpu(prelayer,config.hiddensize,config.return_sequence);
        end
    case 'dense'
        if isfield(config,'loss')
        layer=dense_init_gpu(prelayer,config.hiddensize,config.loss);
        else
            layer=dense_init_gpu(prelayer,config.hiddensize);
        end
    case 'activation'
        if isfield(config,'loss')
        layer=activation_init(prelayer,config.act_fun,config.loss);
        else
        layer=activation_init(prelayer,config.act_fun);
        end
end
end