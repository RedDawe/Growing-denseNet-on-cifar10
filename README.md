# Disclaimer
The dense3n is just an idea. For luck of computational resources I couldn't train or fine-tune the network, or even try the netwok on ImageNet. The ideas should apply though.

Also, probably read CLR, NasNet and denseNet papers before this.

# dense3n

I have always wanted an automatic network, the NasNets were the first (worth mentioning) to try that, but they are extremely expensive to train, so I came up with an alternative way to search for the architecture. While NasNets just basically brute force possible values for all parameters, I decided to handle each one on it's own.

# Structure
The whole network consists of dense block and then reduction block. (from the denseNet paper)

Here's how I handle number of layers in the dense block and dropout rate. Start with a relatively small network with rate zero. You train the network and you look if the CV accuracy (or other metric) is bigger or smaller than the previous model (init with acc = 0), if it is bigger, you can continue so you add more layers to the end of the dense block, if it is lower, you overfited so you increase the rate. Then you train again and repeat.

And why aren't we just training multiple networks? Because every time we increase the rate or add another dense layer, we keep the weights of the network from the previous training, which is allowed by the whole structure of the network, where we're basically taking the network as it was and adding a layer on top of it. (we have to retrain the final reduction block)

The reduction block you want to design as short as possible, so you don't have to retrain as much.

* I used one big dense block, but you can split it into multiple ones with bottlenecks (1x1 convs) after a certain number of layers

# To sum up
Dense block
- your typical dense block, you can introduce bottlenecks and reductions (although, reduction is advised to use only in the begininng)
- at the end of this you add layers
- you reuse the weights here

Reduction block
- instead of using just pooling as in the paper, you do pooling, but at the same time you use convolution of the same size with valid padding and then you concatenate the two
- always has the same number of layers
- you retrain every time


# The rest of the hyper-parameters
  LR - here I just uced the Cyclical Learning Rate
  
  Number of filters - this one is the only one that has to be chosen by you, or you have to run the network multiple times as you would with a NasNet
  
  Concat vs Sum - NasNets worry about this, but denseNets just always use concatenation and they outperform resNets (which use sum), so I just went with pure denseNet here
  
  Kernel size - As shown in the inception-V4 paper, stucking multiple size 3 convolutions usually performs better than any other size (note that in the code, I used 3x3 conv, but it is better to use 1x3 followed by 3x1 convs)
  
  Max vs Avg pool - This was never shown to make much of a difference
  
  Number of epochs, eras (era = training of one network) - You can overcome these, if you set for example to stop training when your accuracy doesn't go up for at least 0.01 by 10 epochs, but I just used hard coded values
 
See the code for the details.

# Blah blah blah

The aim of this project is not to beat the NasNets, neither to be eaisier to train than pure denseNets, but to introduce some middle ground in the expensiveness to train, automation and hopefuly performance. Also, as written in the CLR paper, all of this might seem like you actaully have to pick more parameters than fewer, and that would be true. But the point of doing all of this is that the network should be much more robust to those parameters while still giving better results.

It is motivated by the love for automation and ridiculous demands of NasNets, inspired by denseNet regarding the structure and by CLR paper regarding the automation of the parameters.
