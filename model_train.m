function model=model_train(model,x,y,nb_epoch,verbose)
if nargin<5
    verbose=0;
end
batchsize=model.batchsize;
shape_x=size(x);
shape_y=size(y);
g_batch=1;
nb_batch=floor(shape_x(end)/batchsize)*nb_epoch;
if verbose
h = waitbar(g_batch/nb_batch,'Training model');
end
for epoch=1:nb_epoch
    batch=1;
    tic;
    while batch*batchsize<=shape_x(end)
        %% ff
        if numel(shape_x)==2
            model.layers{1}=x(:,(batch-1)*batchsize+1:batch*batchsize);
        elseif numel(shape_x)==3
            model.layers{1}=x(:,:,(batch-1)*batchsize+1:batch*batchsize);
        else
            error('The number of dims of input data must be 2/3');
        end
        for l=2:length(model.layers)-1
            model.layers{l}=model.layers{l}.ff(model.layers{l},model.layers{l-1});
        end
        %% eval
        if numel(shape_y)==2
        model.layers{end-1}=model.eval_loss(model.layers{end-1},y(:,(batch-1)*batchsize+1:batch*batchsize));
        elseif numel(shape_y)==3
        model.layers{end-1}=model.eval_loss(model.layers{end-1},y(:,:,(batch-1)*batchsize+1:batch*batchsize));
        else
            error('The number of dims of output data must be 2/3');
        end
        loss=model.layers{end-1}.loss(end);
        model.loss=model.layers{end-1}.loss;
        if verbose
        pro=num2str(100*g_batch/nb_batch);
        message=['Training model ','Epoch: ',num2str(epoch),'/',num2str(nb_epoch), ' Progress: ',pro,'%',' loss: ',num2str(loss)];
        waitbar(g_batch/nb_batch,h,message);
        end
        %% bp
        for l=length(model.layers)-1:-1:2
            if model.layers{l}.trainable
                model.layers{l}=model.optimize(model.layers{l}.bp(model.layers{l},model.layers{l+1}),batch,epoch);
            else
                model.layers{l}=model.layers{l}.bp(model.layers{l},model.layers{l+1});
            end
        end
        batch=batch+1;
        g_batch=g_batch+1;
    end
    toc
        if verbose>=2
            plot(model.loss,'r-');
            pause(0.05);
        end
end
delete(h);
end
