function layer=lstm_update(layer,t,opt)
if nargin <= 2
    opt='sgd';
end
switch opt
    case 'sgd'
        if layer.momentum >0
            if t==1
                layer.vW=layer.learningrate*layer.dW;
            else
                layer.vW=layer.momentum*layer.vW-layer.learningrate*layer.dW;
                layer.dW=layer.vW;
            end
            layer.W=layer.W+layer.dW;
        else
            layer.W=layer.W+layer.learningrate*layer.dW;
        end
end
end