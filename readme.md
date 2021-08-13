Here is a brief summary of our efforts in the past 2 months:

1. Data Set:
All our models have been trained on All our models have been trained on [2019+20 merged dataset](https://www.kaggle.com/kingofarmy/cassavapreprocessed).

2. Augmentations:
We experimented with FMix, CutMix and SnapMix. FMix and CutMix didn't perform well in our pipeline, so we dropped them. We plugged SnapMix with linearly increasing probs in almost all our models except ViT and one variation of EfficientNetB3. It enhanced both our CV and public LB especially in ResNeXt variants.

3. Loss Functions:
Since batches are already created using Stratified K-Fold, class imbalance was being dealt with already. And since the test set is itself noisy, noisy loss functions didn't help us out either. So we decided to stick with standard Cross Entropy Loss.

4. Models:
These are the ones that we included in our final ensemble.
efficientnet_b3_ns (w and wo SnapMix), resnext50_32x4d, vit_base_patch16_384, resnest50d.

5. Ensembling:
As long as greed is stronger than compassion, there will always be suffering.

6. Post Processing:
We tried scaling down each class' probabilities with a factor of .975, original idea was to scale 4th class since it has highest representation in train and the 31% test and it didn't work, but surprisingly applying this operation on the 5th class gave an improvement in score, which we later found out was perhaps fluke, and dropped this idea. We didn't try stacking or anything else, since discussions weren't too positive about them.

7. What Failed:

    * Cleaning the dataset.
    * Filtering out lowest 5th or 10th percentile of images based on %age representation of green color.
    * SEResNeXt 50&101 models gave good solo scores but deteriorated ensemble performance miserably.
    * Training on 100% data/ training first 5 out of 10 folds.

8. Acknowledgements:
    * [Pytorch Efficientnet Baseline [Train] AMP+Aug](https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug)
    * [Cassava / resnext50_32x4d starter [training]](https://www.kaggle.com/yasufuminakama/cassava-resnext50-32x4d-starter-training)
    * [SnapMix](https://github.com/Shaoli-Huang/SnapMix)
    * [Epoch thresholding](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/212347)
    