#71.6 Reproduceeing:

- Using mobilenet_config/xgen.json training 300 epoch got best model top1 a round 70.7

- Then using 70.7 as pretrained model, and config  mobilenet_config/tune.json train another 250 epoch got 71.7



# update for mobilenet_config/xgen.json:
- After we reproduce 71.7, we believe  that we can update epoch in xgen.json from 300 to 600, then we can achieve 71.7 train from scratch.
