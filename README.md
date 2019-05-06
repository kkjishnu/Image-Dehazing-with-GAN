# Image Dehaze with GAN

##Dataset

NYU Depth dataset, containing 1449 pairs of hazy and ground truth images. The dataset can be downloaded and extracted by the following command. It will be placed into folders `A` and `B`.
```
wget -O data.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python3 extract.py
```
Split the dataset into train and test and place them in a folder 

├── datasets                   
    |   ├── Haze2Dehaze        
    |   |   ├── train              # Training
    |   |   |   ├── A              
    |   |   |   └── B              
    |   |   └── test               # Testing
    |   |   |   ├── A              
    |   |   |   └── B              