# Contrastive-Learning
For this project I have done two experiments.

1. To implement SimCLR framework and to use a simple image classification task on STL-10 dataset. My objective was to prove the significance of
   contrastive learning and to study the components of the SimCLR architecture, especially the removal of projection head during downstream task,
   training for longer training time etc.
2. For my second experiment, I re-implemented the supervised contrastive loss (CIFAR-10 dataset),which the authors claim is better 
   than the cross entropy loss. 


I have trained the model and uploaded to my google drive, those pretrained models can be used for getting results, because training the models from
scratch will take lot of time. 
link to my googledrive : https://drive.google.com/drive/folders/1rn5Fd4oYzZtvjwr3p4HM7kM1DKbfaUZU?usp=share_link



repositories used: 1. https://github.com/HobbitLong/SupContrast
                   2. https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb

Code used from the repositories: The simclr framework was highly inspired from the 2nd repository mentioned above. I made modifications in it
for implementing it in a simpler form. I modified the environment and trained the model for 500 and 100 epochs rather than using the pretrained models 
present in the repository. Training the model for 500 epoch (on STL10 dataset) took plenty of hours, so i would suggest to use the pre-trained model from my google drive. 
I also tried running the CIFAR-10 for simclr, but the results were not satisfactory. For the second experiment, I re-implemented the code from the first
repository mentioned above. But I did only the supervised contrastive learning, for CIFAR-10, for 100 epochs, to get the results and compare with 
the results in the paper. I tried running the simclr setting using this code, but it was not giving me valuable results. 
Code done by me: I came up with the testing for verifying that using the projection head for the downstream task can cause a decrease in accuracy. Thus, 
I proceeded the downstream task with projection head and got an accuracy of 76.96% for the full dataset, which was close to 5% less than the accuracy
without the projection head.


Datasets: I used the STL-10 dataset for first experiment and CIFAR-10 dataset for the second experiment, both the datasets are available in torchvision.



How to run the code:
. For experiment 1,
. Firstly, run the simclr.py file, which will start the training of the model, but will also create a folder named: saved_models. Download the files
  from the google drive link and paste it in this folder, so that you can use the pretrained model.  
. for using the model trained for 500 epochs, run the simclr.py file by:
	python simclr.py --epochs 500 ; the downstream task of image classification is performed and an accuracy of 80.06% is obtained 
. For using the model trained for 100 epochs, run the simclr.py file by:
	python simclr.py --epochs 100 ; the downstream task of image classification is performed and an accuracy of 75.49% is obtained
. For using the framework with projection head, run the simclr_with_projection_head.py file; this will show the accuracy for the downstream task
  when the projection head is used, the accuracy value will less than the method without projection head. 

. For the second experiment I re-implemented the code from the second repository mentioned in the paper. 
. For pre-training, run the main_supcon.py file by:
      python main_supcon.py --batch_size 1024 \
        --learning_rate 0.5 \
        --temp 0.1 \
        --cosine

. For linear evaluation,  run the main_linear.py file by:
      python main_linear.py --batch_size 512 \
        --learning_rate 5 \
        --ckpt /path/to/model.pth

SimCLR architecture:
. The image first undergoes data augmentation and two versions of the same image is created after this transformation.
. The transformed images are forwareded to encoder networks.
. The base encoder is a cnn, and coverts the image to vector representations.
. Further the representations are fed to projection head where they undergo non linear transformation, this is multi layer perceptron, with
  two dense layers and ReLu activation.
. The loss function is applied to the representations from this stage. Projection head is used only for pre-training stage and base encoder is
  used as the feature extractor for downstream tasks. For my first experiment, the loss function used was NT-Xent, normalised temperature scaled 
  cross entropy loss, and in the second experiment, the main difference was in the use of loss function, a new loss function called the
  supervised contrastive loss was used, which the authors claim is better than the cross entropy loss. 
