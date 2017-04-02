function dx=dact(y,fun)
switch fun
    case 'sigmoid'
        dx = y .* (1 - y);
        return
    case 'tanh'
        dx=1-tanh(y).^2;
        return
    case 'Relu'
        return
    case 'linear'
        dx = y;
        return
    case  'softmax'
        dx=y;
        return
end      
end