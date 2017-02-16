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
