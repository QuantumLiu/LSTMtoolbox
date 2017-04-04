function outputlayer=eval_loss(outputlayer,y_true)
num=numel(y_true);
loss=feval(@(x)mean(x(:)),outputlayer.loss_f(single(y_true),outputlayer.output,num));
outputlayer.loss=[outputlayer.loss,loss];
if isequal(outputlayer.type,'lstm')&& ~outputlayer.return_sequence
outputlayer.e(:,end,:)=-loss.*outputlayer.loss_df(y_true,outputlayer.output,num);
else
outputlayer.e=-loss.*outputlayer.loss_df(single(y_true),outputlayer.output,num);
end
end
