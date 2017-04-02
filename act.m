function y=act(x,fun)
switch fun
    case 'sigmoid'
        y = 1./(1+exp(-x));
        return
    case 'tanh'
        y=tanh(x);
        return
    case 'softmax'
        E=exp(x- max(x,[],1));
        y =  E./ sum(E,1) ;
        return
    case 'Relu'
        %y = max(x, single(0)) ;
        y=x.*(x>0);
        return
end