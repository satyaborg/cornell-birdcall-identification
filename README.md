# Cornell Birdcall Identification 🐦

Repository for Kaggle kernels, training pipeline and everything else (except raw audio data!)

### TODO

#### Main tasks:
- [x] Implement training and validation pipeline
- [x] Implement submission notebook
- [x] Implement: SimpleCNN (full clip), ConvNet (5 sec segments)
- [x] Random crop 5 seconds per clip during training
- [ ] Implement Bidirectional LSTM
- [ ] Extract secondary labels and use for training
- [ ] Augmentations: [SpecAugment](https://github.com/DemisEom/SpecAugment) + MixUp
- [ ] Standard sound augmentation for raw inputs to LSTM (pitch shift etc)
- [ ] Add Attention and skip connections to CNN
- [ ] Additional metric: mAP

#### Auxillary tasks:
- [ ] Use [VGGish](https://github.com/harritaylor/torchvggish) (trained on [Audioset](https://research.google.com/audioset/)) to extract 128-dimensional embedding from raw audio clips (outputs @~1 sec)
- [ ] Encode phylogeneric information
- [ ] Create a co-occurrence matrix of birds based on location

### Papers
- [Bird Identification from Timestamped,Geotagged Audio Recordings](http://ceur-ws.org/Vol-2125/paper_181.pdf)
- [Large-Scale Bird Sound Classification using Convolutional Neural Networks](http://ceur-ws.org/Vol-1866/paper_143.pdf)
- [Bird Sound Recognition Using a Convolutional Neural Network](https://www.researchgate.net/publication/328836649_Bird_Sound_Recognition_Using_a_Convolutional_Neural_Network)

### Reference code
- https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training
