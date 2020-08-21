
### (..)[https://www.researchgate.net/publication/328836649_Bird_Sound_Recognition_Using_a_Convolutional_Neural_Network]
The winner of the 2016 BirdCLEF challenge, Sprengel et
al. [9] (68.6% for single labeling and 55.5% for multiple
labeling) discuss classiﬁcation with CNN containing ﬁve con-
volutional and one dense layer. As input, spectrograms are
generated from the audio ﬁles after splitting the noise from
the actual bird sound

E. Sprengel, M. Jaggi, Y. Kilcher, and T. Hofmann, “Audio based bird
species identiﬁcation using deep learning techniques,” Working notes of
CLEF, 2016

- MobileNet pre-trained CNN
- Advantage: if color map of the spectograms matches the pre-trained model's training images

STFT: To this end, we use spectrograms, which are a
visual representation of the magnitude returned by the Short
Time Fourier Transform (STFT)[4]. STFT is a version of
the Discrete Fourier Transform (DFT), which instead of only
performing one DFT on a longer signal, splits the signal into
partially overlapping chunks and performs the DFT on each
using a sliding window.

His experiments involve networks trained from scratch, mel-
scaled power spectrograms, an upper frequency cap and noise
ﬁltering.

As input, spectrograms are
generated from the audio ﬁles after splitting the noise from
the actual bird sound

In his research involving template matching and
bagging, Lasseck [11] proposes spectrogram downsampling
as a performance improvement without signiﬁcant detriment
to the quality

split audio files to chunk

## Data
- Use > 3.0 rating files
- Lower the rating, noisier the labels
- Start with 100 classes per species (if available)

## Pretrained models
- VGGish

## Augmentation
- SpecAugment: 3 main types from Park et al. introduced [SpecAugment](https://arxiv.org/abs/1904.08779) for data augmentation in speech recognition. There are 3 basic ways to augment data which are time warping, frequency masking and time masking.
