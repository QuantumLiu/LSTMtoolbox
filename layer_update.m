function layer=layer_update(layer,pars)
batch=layer.batch;
if nargin <2
    pars.opt='sgd';
end
switch pars.opt
    case 'sgd'
        if pars.momentum >0
            if batch==1
                layer.vW=pars.learningrate*layer.dW;
            else
                layer.vW=pars.momentum*layer.vW-pars.learningrate*layer.dW;
                layer.dW=layer.vW;
            end
            layer.W=layer.W+layer.dW;
        else
            layer.W=layer.W+pars.learningrate*layer.dW;
        end
end
end