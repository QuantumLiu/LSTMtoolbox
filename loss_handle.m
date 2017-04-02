function [f,df]=loss_handle(type)
syms y_true y_pred num
switch type
    case 'mse'
        symsf(y_true,y_pred,num)=((y_true-y_pred).^2)/num;
        f=matlabFunction(symsf);
        df=matlabFunction(diff(symsf,y_pred));
        return
    case 'cross_entropy'
        symsf(y_true,y_pred,num)=(-1/num).*sum(y_true.*(y_pred)+(1-y_true).*log(1-y_pred));
        f=matlabFunction(symsf);
        df=matlabFunction(diff(symsf,y_pred));
        return   
    case 'categorical_cross_entropy'
        symsf(y_true,y_pred,num)=(-1/num).*(log(y_true.*y_pred+0.0001));
        f=matlabFunction(symsf);
        df=matlabFunction(diff(symsf,y_pred));
end
end
