# Grounding 3D Object Affordance from 2D Interactions in Images

PyTorch implementation of Grounding 3D Object Affordance from 2D Interactios in Images. This repository contains PyTorch training and evaluation code, the dataset will coming soon.

## 📋 Table of content
 1. [📎 Paper Link](#1)
 2. [💡 Abstract](#2)
 3. [📖 Method](#3)
 4. [📂 Dataset](#4)
 5. [📃 Requirements](#5)
 6. [✏️ Usage](#6)
    1. [Train](#61)
    2. [Evaluate](#62)
    3. [Inference](#63)
 7. [📊 Experimental Results](#7)
 8. [🍎 Potential Applications](#8)
 9. [✉️ Statement](#9)
 10. [🔍 Citation](#10)

## 📎 Paper Link <a name="1"></a> 
* Grounding 3D Object Affordance from 2D Interactions in Images ([link](https://arxiv.org/abs/2106.14747))
> Authors:
> Yuhang Yang, Wei Zhai, Hongchen Luo, Yang Cao, Jiebo Luo, Zhengjun Zha

## 💡 Abstract <a name="2"></a> 
Grounding 3D object affordance seeks to locate objects' ''action possibilities'' regions in the 3D space, which serves as a link between perception and operation for embodied agents. Existing studies primarily focus on connecting visual affordances with geometry structures, e.g. relying on annotations to declare interactive regions of interest on the object and establishing a mapping between the regions and affordances. However, the essence of learning object affordance is to understand how to use it, and the manner that detaches interactions is limited in generalization. Normally, humans possess the ability to perceive object affordances in the physical world through demonstration images or videos. Motivated by this, we introduce a novel task setting: grounding 3D object affordance from 2D interactions in images, which faces the challenge of anticipating affordance through interactions of different sources. To address this problem, we devise a novel Interaction-driven 3D Affordance Grounding Network (IAG), which aligns the region feature of objects from different sources and models the interactive contexts for 3D object affordance grounding. Besides, we collect a Point-Image Affordance Dataset (PIAD) to support the proposed task. Comprehensive experiments on PIAD demonstrate the reliability of the proposed task and the superiority of our method. The dataset and code will be made available to the public. 

<p align="center">
    <img src="./img/fig1.png" width="500"/> <br />
    <em> 
    </em>
</p>

**Illustration of perceiving affordance.** We propose to ground 3D object affordance through 2D interactions. Inputting an object point cloud with an interactive image, grounding the corresponding affordance on the 3D object.

## 📖 Method <a name="3"></a> 
### IAG-Net <a name="31"></a> 
<p align="center">
    <img src="./img/pipeline.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Our Interaction-driven 3D Affordance Grounding Network.** it firstly extracts localized features $F_{i}$, $F_{p}$ respectively, then takes the Joint Region Alignment Module to align them and get the joint feature $F_{j}$. Next, Affordance Revealed Module utilizes $F_{j}$ to reveal affordance $F_{\alpha}$ with $F_{s}$, $F_{e}$ by cross-attention. Eventually, $F_{j}$ and $F_{\alpha}$ are sent to the decoder to obtain the final results $\hat{\phi}$ and $\hat{y}$.

## 📂 Dataset <a name="4"></a> 
<p align="center">
    <img src="./img/dataset.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Properties of the PIAD dataset.** **(a)** Data pairs in the PIAD, the red region in point clouds is the affordance annotation. **(b)** Distribution of the image data. The horizontal axis represents the category of affordance, the vertical axis represents quantity, and different colors represent different objects. **(c)** Distribution of the point cloud data. **(d)** The ratio of images and point clouds in each affordance class. It shows that images and point clouds are not fixed one-to-one pairing, they can form multiple pairs.

<p align="center">
    <img src="./img/data_sample.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Examples of PIAD.** Some paired images and point clouds in PIAD. The ''yellow'' box in the image is the bounding box of the interactive subject, the ''red'' box is the bounding box of the interactive object.

```bash  
We will release our PIAD soon...
```


## 📃 Requirements <a name="5"></a> 
  - python-3.9 
  - pytorch-1.13.1
  - torchvision-0.14.1
  - open3d-0.16.0
  - scipy-1.10.0
  - matplotlib-3.6.3
  - numpy-1.24.1
  - OpenEXR-1.3.9
  - scikit-learn-1.2.0
  - mitsuba-3.0.1

## ✏️ Usage <a name="6"></a> 

```bash  
git clone https://github.com/yyvhang/IAGNet.git
```

### Download PIAD <a name="41"></a> 
- We will release the PIAD dataset soon.



### Train <a name="61"></a> 
To train the IAG-Net model, you can modify the training parameter in `config/config_seen.yaml` and then run the following command:
```bash  
python train.py --name IAG
```
To train the IAG-Net model by DDP, run the `train_DDP.py` as follows:
```bash  
python -m torch.distributed.run --nproc_per_node=4 train_DDP.py  --gpu_num 4  --name IAG_DDP  
```
`gpu_num, nproc_per_node` is the count of GPU, chose it according to your devices.

### Evaluate <a name="62"></a> 
To evaluate the trained IAG-Net model, run `evalization.py`:
```bash  
python evalization.py
```
### Inference <a name="63"></a> 
To inference the results with IAG-Net model, run `inference.py` to get the `.ply` file
```bash  
python inference.py --model_path runs/train/IAG/best.pt
```

To render the `.ply` file, we provide the script `rend_point.py`, run this file to get `.xml` files:
```bash  
python rend_point.py
```
Once you get the `.xml` file, just rend it with `mitsuba`:
```bash  
mitsuba Chair.xml
```

## 📊 Experimental Results <a name="7"></a> 
- Visual results of comparison in both partitions:
<p align="center">
    <img src="./img/visual_compare.png" width="800"/> <br />
    <em> 
    </em>
</p>

- One image with multiple point clouds:
<p align="center">
    <img src="./img/i2p.png" width="750"/> <br />
    <em> 
    </em>
</p>

**(a)** Same object category. **(b)** Different object categories, similar geometrics. **(c)** Different object categories and geometrics.

- One point cloud with different images:
<p align="center">
    <img src="./img/p2i.png" width="750"/> <br />
    <em> 
    </em>
</p>

Grounding affordance on the **same** point cloud with images that contain **similar** or **disparate** interactions

- Rotate and partial point cloud:
<p align="center">
    <img src="./img/rotate.png" width="750"/> <br />
    <em> 
    </em>
</p>

Results on the **rotate** and **partial** point clouds.


## 🍎 Potential Applications <a name="8"></a> 

<p align="center">
    <img src="./img/application.png" width="650"/> <br />
    <em> 
    </em>
</p>

**Potential Applications of IAG affordance system.** This work has the potential to bridge the gap between perception and operation, serving areas like demonstration learning, robot manipulation, and may be a part of human-assistant agent system e.g. Tesla Bot, Boston Dynamics Atlas.

## ✉️ Statement <a name="9"></a> 
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [yyuhang@mail.ustc.edu.cn](yyuhang@mail.ustc.edu.cn).

## 🔍 Citation <a name="10"></a> 

```
@inproceedings{Oneluo,
  title={One-Shot Affordance Detection},
  author={Hongchen Luo and Wei Zhai and Jing Zhang and Yang Cao and Dacheng Tao},
  booktitle={IJCAI},
  year={2021}
}
```

