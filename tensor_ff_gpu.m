function layer=tensor_ff_gpu(layer,input)
if strcmpi(layer.type,'input')
    layer.output=input;
else
    layer.output=input.output;
end
end