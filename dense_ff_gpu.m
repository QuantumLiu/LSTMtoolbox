function layer=dense_ff_gpu(layer,prelayer)
if isequal(class(prelayer),'struct')
    if ~isequal(size(prelayer.output),layer.input_shape)
        error('Shape unmatched!')
    end
    layer.input(1:end-1,:)=prelayer.output;
else
    layer.input(1:end-1,:)=prelayer;
end
layer.output=layer.W*layer.input;
end
