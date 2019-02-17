# dense3n

The dense3n is just an idea, how to take transfer learning on another level. For luck of computational resources I couldn't train or fine-tune the network, or even try the netwok on ImageNet. The ideas should apply though.

It is inspired by denseNet regarding the structure and by CLR paper regarding the automation of the parameters, while being motivated by the ridiculous demands of NasNets. But considering that NasNets are extremely expensive to train,
I came up with an alternative way to search for the architecture. While nasnets just basically brute force possible values for all parameters, I decided to handle each one
on it's own. As I have said, the basic structure is denseNet, first with dense block(s) and then with a special dense/reduction block.

The parameters go as follows:
  
  LR - here I just uced the Cyclical Learning Rate
  
  Number of filters - this one is the only one that has to be chosen by you, or you have to run the network multiple times as in NasNet
  
  Concat vs Sum - NasNets worry about this, but denseNets just always use concatenation and they outperform resNets (which use sum), so I just went with pure denseNet here
  
  Kernel size - As shown in the inception-V4 paper, stucking multiple size 3 convolutions performs usually better than any other size (note that in the code, I used 3x3 conv, but it is better to use 1x3 followed by 3x1 convs)
  
  Max vs Avg pool - This was never shown to make much of a differnce
  
  Number of epochs, eras (era equals one training of the network) - You can overcome these, if you set for example to stop training when your accuracy doesn't go up for at least 0.01 by 10 epochs, but I just used hard coded values
  
  Dropout rate, number of layers - This is the main idea here. You start with a relatively small network with rate zero. You train the network and you look if the CV accuracy (or other metric) is bigger or smaller than the previous model (init with acc = 0), if it is bigger, you can continoue so you add more layers, if it is lower, you overfited so you increase the rate.
                                   
See the code for the details.

The aim of this project is not to beat the NasNets, neither to be eaisier to train than pure denseNets, but to introduce some middle ground with the expensivenes to train, automation and hopefuly performance. Also, as written in the CLR paper, all of this might seem like you actaully have to pick more parameters than fewer, and that would be true. But the point of doing all of this is that the network should be much more robust to those parameters while still giving better results.
