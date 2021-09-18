## RetinexDIP
The pytorch implementation of [RetinexDIP: A Unified Deep Framework for Low-light Image Enhancement.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9050555)

>Z. Zhao, B. Xiong, L. Wang, Q. Ou, L. Yu and F. Kuang, "RetinexDIP: A Unified Deep Framework for Low-light Image Enhancement," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2021.3073371.

## Install
- scipy==1.2.1
- numpy==1.19.4
- opencv-python==4.1.1
- Pillow==8.1.2
- torch==1.2.0
- torchvision==0.4.0

## Files Structure
├─data
│  ├─test
├─net
├─output
│  ├─illumination
│  └─reflection
├─result
└─utils

## Dataset
- [DICM](http://mcl.korea.ac.kr/projects/LDR/LDR_TEST_IMAGES_DICM.zip)
- [ExDark](http://web.fsktm.um.edu.my/~cschan/source/CVIU/ExDark.zip)
- [LIME](http://cs.tju.edu.cn/orgs/vision/~xguo/LIME.htm)
- [NASA](http://dragon.larc.nasa.gov/retinex/pao/news/)
- [VV](https://sites.google.com/site/vonikakis/datasets)
- *Fusion*, X. Fu, D. Zeng, Y. Huang, Y. Liao, X. Ding, and J. Paisley, ”A fusion-based enhancing method for weakly illuminated images,” Signal Processing, vol. 129, pp. 82-96, 2016.
- *NPE*, S. Wang, J. Zheng, H. Hu, and B. Li, ”Naturalness preserved enhancement algorithm for non-uniform illumination images,” IEEE Transactions on Image Processing, vol. 22, pp. 3538-3548, 2013

## Experiments
```
python Retinexdip.py --input data/test --result ./result
```
Before  running the code, you must assure that  every dataset is included in  the input root directory. For example, these datasets should be included in the "./data/test":
```
datasets = ['DICM', 'ExDark', 'Fusion', 'LIME', 'NPEA', 'Nasa', 'VV']
```
Explanations for some hyperparameters:
- **input_depth**
  This value could affect the performance. 3 is ok for natural image, if your images are extremely dark, you may consider 8 for the value.

- **flag**

  This parameter from the function named $get_enhanced$ can be set as $True$ and $False$. If the input image is extremely dark, setting the flag as True can produce promising result. 

