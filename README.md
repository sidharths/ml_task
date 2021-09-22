# ml_task

## What the problem is about? Dataset used, and associated challenges

Here we are trying to do image classification of a very imbalanced dataset. There are three classes in percentages
* FULL_VISIBILITY  -  80.75%
* PARTIAL_VISIBILITY  -  15.375%
* NO_VISIBILITY   -   3.875%

The problem essentially is in developing a model that can learn despite the class imbalance.
A naive model that describes every sample as *FULL_VISIBILITY* will get an overall accuracy of 80.75%
While the assignment wants an overall accuracy greater than 82% we would need to observe each classes performance to have a generalizable model.

some challenges:

1. Due to the class imbalance the split between train, val and test is critical. val set is used to pick the best model during training of several epochs. We want the split sets to be representative of the overall distribution of classes
2. Due to the approach to have each split with same propotion of classes as overall distribution we end up with a validation and test set of 104 sample. The breakdown of which is **84 VISIBLE, 16 PARTIAL, 4 NOT VISIBLE**. The minority classes are very small (*16 and 4 in size*) in test set and the model performance is sensitive to the "random" factor htat sent what kind of samples (easy or edge cases) into the split.
3. There are different approaches to deal with class imbalance. One way would be to oversample minority classes during training. Another approach will be to give relative weights of classes to the loss function. So that minority classes are contributing more to the loss.
4. Overtraining and getting a high training accuracy that is not reflected in validation or test set.



## Describe your proposed solution

Plan to work on this assignment using pytorch framework

* stratified split dataset into train, test, val (maintains class ratios in each of the three)
* Use a lightweigth pre trained DNN and fine tune with our dataset
* Planning to use resnet-18 from pytorch model zoo with pre trained weights
* adding a final dense layer of 512 input channels and 3 output channels for  the 3 classes
* Do some image augmentations while training (random horizontal flip random vertical flip). Have to be careful while picking augmentations. Random crop for example cannot be done because it can potentially change the label of the sample for ex from FULL_VISIBLE to partially or not visible.
* Use albumentations for various image augmentations but carefully avoid any augmentation that can crop the image or transfer data from one channel to other (like A.ChannelShuffle())
* use nn.CrossEntropyLoss with weights of class to amplify loss contribution of minor classes
* use a learning rate scheduler to stabilize the model for later epochs

## Future works and additional potential approaches to tackle the given problem.

1. Try specialized loss functions like focal loss can be used which deal better than weighted classes in CrossEntropyLoss https://amaarora.github.io/2020/06/29/FocalLoss.html
2. While model will be sensitive to the initial data split between (train+val) and (test). For minority classes if the "hard" cases all end up in test set then model will not be able to train well to generalize. However we can remove some of those issues for the (train+val) set by doing k-fold crossvalidation on them.
3. As part of EDA I would like to eye ball all the samples and check if they are labelled correctly. Some of the cases might be labelled incorrectly.
4. Definitely collecting more data for the minority classes will be a realistic solution if it is possible to do so.
5. If it was a truly noisy image (instead of 1 clean channel + 2 gaussian white noise channels) then we could have tried some form of image denoising network https://kornia.readthedocs.io/en/v0.5.0/tutorials/total_variation_denoising.html  I have not looked into this in detail. Perhaps the output images from denoised can be connected with image classifier.
6. Try out ImbalancedDatasetSampler https://github.com/ufoym/imbalanced-dataset-sampler. I dont understand it well but results they show seem promising.