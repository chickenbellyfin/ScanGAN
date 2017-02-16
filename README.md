# ScanGAN

Convenient GAN implementation in [Keras](http://keras.io)

This is a WIP

## Basic Usage
```python
from GAN import GAN

# 128x128, 3 channels

shape = (128, 128, 3)
model = GAN(shape)

model.train(x, y, epochs=100)

pred = model.generate(x[0])
```

## Settings
You may construct the model with some custom hyperparameters by passing in a *settinng* object at construction.

```python
from keras.optimizers import Adagrad

shape = (64, 64, 1) # 64x64 grayscal images

my_settings = {
	'input_mask':True,
	'g_ksize': 3,
	'd_optimizer': Adagrad()
}

model = GAN(shape, settings=my_settings)
```

|Key|Type|Default Value|Description|
|input_mask|boolean|False| If set to ```True```, the model with replace the output of the generator with the input where the input != 0. The masked output is used when computing loss|
|d_loss_target|0.3|float between 0 ... 1|Each epoch, the discriminator will be trained if its loss is less than the target, the generator will be trained otherwise|
|g_optimizer|str, keras optimizer, or lambda|Adam(1e-3)|The optimizer that will be used for the Generator.|
|g_ksize|int|5|Convolution kernel size for the generator (g_ksize * g_ksize)|
|g_depth|int|64|Output depth of each hidden layer of the generator|
|g_activation|str or lambda|lambda: LeakyReLU()|Activation function to use for the generator
|g_regularizer|str of lambda|None|Regularization to use for the generator
|d_optimizer|str, keras optimizer, or lambda|SGD()|The optimizer that will be used for the Discriminator.|
|d_ksize|int|5|Convolution kernel size for the discriminator (d_ksize * d_ksize)|
|d_depth|int|32|Output depth of each hidden layer of the discriminator|
|d_activation|str or lambda|lambda: LeakyReLU()|Activation function to use for the discriminator
|d_output_activation|str or lambda|'sigmoid'|Activation function to use for *output layer* the discriminator
|d_regularizer|str of lambda|None|Regularization to use for the discriminator

## Sample Output (inpainting)

* Training set was ~400 128x128 RGB images of landscapes
* Default settings
* epochs = 100

![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/training.png)


### Input / Output / Original

![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex0.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex1.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex2.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex3.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex4.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex5.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex6.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex7.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex8.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex9.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex10.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex11.png)
![](https://github.com/chickenbellyfinn/ScanGAN/raw/master/output/ex12.png)
