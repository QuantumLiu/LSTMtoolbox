#LSTMtoolbox
One of the fastest matlab's RNN libs.
Performance:
=
model:A LSTM model has [1024,1024,1024] hidensizes and 10 
timestep with a 256 dims input.
+++++++++++++++++++++++++++++
Device: i7-4710hq,GTX940m
+++++++++++++++++++++++++++++
LSTMtoolbox: 60sec/epoch 
+++++++++++++++++++++++++++++
Keras(1.2.2,Tensorflow backend,cudnn5.1): 29sec/epoch
Features
=
High parallel Implementation.

 - Concatance the weights of 4 gates to **W** and the values of **x** and **h** of every timesteps in a batch to a 3D tensor **xh**.Compute **x*W** for every timesteps of every samples in a batch at one time.
 - Compute the activated values of **input,forget ,ouput gates** at one time.

OOP style

 - Use `struct` type to define a **layer** class.

	 