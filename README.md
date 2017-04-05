# LSTMtoolbox
One of the fastest matlab's RNN libs.
## Performance
model:A LSTM model has [1024,1024,1024] hidensizes and 10 
timestep with a 256 dims input.  
Device: i7-4710hq,GTX940m  
LSTMtoolbox: 60sec/epoch Keras(1.2.2,Tensorflow backend,cudnn5.1): 29sec/epoch 
## Features
High parallel Implementation.


* Concatance the weights of 4 gates to **W** and the values of **x** and **h** of every timesteps in a batch to a 3D tensor **xh**.Compute **x*W** for every timesteps of every samples in a batch at one time.
* Compute the activated values of **input,forget ,ouput gates** at one time.

OOP style
* Use `struct` type to define a **layer** class and a **model** class.Define **ff**, **bp**, **optimize** methods by using a `FunctionHandle`.  
## Model
* A `model` is a set of `layers`,`data` and `optimizer`.
* `model=model_init(input_shape,configs,optimizer)`
    * `input_shape` : a `vector`,`[input_dim,batchsize]` or `[input_dim,timestep,batchsize]`
    * `configs` : `cell` ,configures of each layers
    * `optimizer` : `struct` ,keywords: `opt`(type of optimizer) ,`learningrate`
    * **example**:  
input_shape=[100,10,64];  
hiddensize=[512,512,512];  
for l=1:length(hiddensize)  
    configs{l}.type='lstm';  
    configs{l}.hiddensize=hiddensize(l);  
    configs{l}.return_sequence=1;  
end  
configs{l+1}.type='activation';  
configs{l+1}.act_fun='softmax';  
configs{l+1}.loss='categorical_cross_entropy';  
optimizer.learningrate=0.1;  
optimizer.momentum=0.2;  
optimizer.opt='sgd';
model=model_init(input_shape,configs,optimizer);  
    

## Layers
### Layer class: 
* attributes:  
    * `type` : `string`,type of the layer,available types:`input`,`dense`,`lstm`,`activation`  
    * `prelayer_type` : `string`,type of the previous layer,available types:`input`,`dense`,`lstm`,`activation`
    * `trainable` : `bool`,is the layer trainable
    * `input_shape` : a `vector`,`[input_dim,batchsize]` or `[input_dim,timestep,batchsize]`
    * `output_shape` : a `vector`,`[hiddensize,batchsize]`or`[hiddensize,timestep,batchsize]`
    * `batch` : `int`,how many batches have been passed
    * `epoch` : same to `batch`
* methods:  
    * `layer=layer_init(prelayer,loss,kwgrs)`
        * Built and init a layer.If the layer is a `input` layer,`prelayer` argument should be `input_shape`
    * `layer=layer.ff(layer,prelayer)`
    * `layer=layer.bp(layer,nextlayer)`  
    #### LSTM layer(layer)

	 