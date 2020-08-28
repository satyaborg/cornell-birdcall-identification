# Cornell Birdcall Identification 🐦

Repository for Kaggle kernels, training pipeline and everything else (except raw audio data!)

### Run
Refer to `config.yaml` for required arguments and hyperparmeters.
Execute the following to train model:
```python
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python -m train
```

### TODO

#### Main tasks:
- [x] Preprocess data: `.mp3` to mel-spectograms (saved as `.npy`) resampled @32KHz
- [x] Implement training and validation pipeline
- [x] Implement submission notebook
- [x] Stratified KFold
- [x] Implement SimpleCNN (full clip), ConvNet (5 sec segments)
- [x] Random crop 5 seconds per clip during training
- [x] [SpecAugment](https://arxiv.org/abs/1904.08779)
- [x] Extend/fill clips with duration less than 5 sec (experimental!)
- [ ] [VGG16/VGG19](https://pytorch.org/hub/pytorch_vision_vgg/)
- [ ] Test time augmentation
- [ ] Handle class imbalance
- [ ] Threshold moving (current default=0.5)?
- [ ] Mixup
- [ ] Implement Recurrent ConvNet / ConvLSTM OR Bidirectional LSTM
- [ ] Extract secondary labels and use for training
- [ ] Add Attention and skip connections to CNN
- [ ] Additional metric: mAP

#### Auxillary tasks:
- [ ] Use pretrained ImageNet models (VGG, ResNet, DenseNet etc) for finetuning
- [ ] Use [VGGish](https://github.com/harritaylor/torchvggish) (trained on [Audioset](https://research.google.com/audioset/)) to extract 128-dimensional embedding from raw audio clips (outputs @~1 sec)
- [ ] Encode phylogeneric information
- [ ] Create a co-occurrence matrix of birds based on location
- [ ] [Mixed precision training](https://pytorch.org/docs/stable/notes/amp_examples.html): `fp32` to `fp16`
- [ ] Port code to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ ] Add [Optuna](https://github.com/optuna/optuna) for hyperparameter tuning; Also check [Ray](https://github.com/ray-project/ray)
- [ ] Standard sound augmentation for raw inputs to LSTM (pitch shift etc)
- [ ] Sound Event Detection (SED)

### Papers
- [Bird Identification from Timestamped,Geotagged Audio Recordings](http://ceur-ws.org/Vol-2125/paper_181.pdf)
- [Large-Scale Bird Sound Classification using Convolutional Neural Networks](http://ceur-ws.org/Vol-1866/paper_143.pdf)
- [Bird Sound Recognition Using a Convolutional Neural Network](https://www.researchgate.net/publication/328836649_Bird_Sound_Recognition_Using_a_Convolutional_Neural_Network)

### Reference code
- https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training
