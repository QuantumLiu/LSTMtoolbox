function layer =dense_bp_gpu(layer,next_layer)
if isequal(class(next_layer),'struct')
    if ~isequal(size(next_layer.dx),layer.output_shape)
        error('Shape unmatched!')
    end
    layer.e=next_layer.dx;
end
layer.dW=layer.e*layer.input';
if ~isequal(layer.prelayer_type,'input')
    layer.dx=layer.W(:,1:end-1)'*layer.e;
end
end