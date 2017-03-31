function layer=lstm_init_gpu(dim,timestep,batchsize,hiddensize,opts)
%% Basic layer attributes
%Input tensor sahpe
layer.input_shape=[dim,timestep,batchsize];
%Output tensor shape
layer.output_shape=[hiddensize,timestep,batchsize];
%The type of the layer
layer.type='lstm';
%conected layer type
layer.conected_type=opts.conected_type;
%The hiddensize of the layer
layer.hiddensize=hiddensize;
%% lstm layer attributes
%Timestep 
layer.timestep=timestep;
layer.batchsize=batchsize;
%n is the number of unrolled timesteps in one batch
layer.n=batchsize*timestep;
%Put x(t) and h(t) in one array 
layer.xh=zeros([dim+1+hiddensize,timestep+1,batchsize],'single','gpuArray');
r_h=dim+1+(1:hiddensize);
%The input tensor
layer.input=zeros(layer.input_shape,'single','gpuArray');
%The output tensor
layer.output=layer.xh(r_h,2:end,:);
%W is the weights of all four gates and bias
layer.weights_dim=dim+1+hiddensize;
layer.W=rand([4*hiddensize,layer.weights_dim],'single','gpuArray')-0.5;
%Compute the value of x_t*wx_t for all ts in one time
layer.maX=zeros([4*hiddensize,timestep,batchsize],'single','gpuArray');
%value before activited
layer.ma=layer.maX;
%value activited
layer.mb=layer.maX;
%sc:state of cell
layer.sc=zeros([hiddensize,timestep,batchsize],'single','gpuArray');
layer.bc=layer.sc;
%diffs
layer.dW=zeros(size(layer.W),'single','gpuArray');
layer.dma=zeros([4*hiddensize,timestep+1,batchsize],'single','gpuArray');
layer.dmb=layer.dma;
layer.dsc=layer.sc;
layer.dh=layer.dsc;
if layer.conected_type~='input'
    layer.dx=layer.input;
end
layer.vW=layer.dW;

%% methods
layer.act_f =@(x)act(x,'sigmoid'); % active function for gate
layer.act_tc =@(x)act(x, 'tanh'); % active function for tc
layer.act_h = @(x)act(x, 'tanh');
layer.dact_f= @(x)dact(x,'sigmoid');
layer.dact_tc =@(x)dact(x, 'tanh'); % active function for tc
layer.dact_h = @(x)dact(x, 'tanh');
layer.learningrate = opts.learningrate;
layer.momentum = opts.momentum;
end