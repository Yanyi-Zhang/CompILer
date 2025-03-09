# Not Just Object, But State: Compositional Incremental Learning without Forgetting (NeurIPS 2024)

Hey there!

This is PyTorch code for the NeurIPS 2024 paper:
**Not Just Object, But State: Compositional Incremental Learning without Forgetting**
*Yanyi Zhang, Binglin Qiu, Qi Jia, Yu Liu, Ran He*
NeurIPS 2024, the Thirty-Eighth Annual Conference on Neural Information Processing Systems

## Environment

The system I used and tested in

- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage

First, clone the repository locally:

```
git clone https://github.com/Yanyi-Zhang/CompILer
cd CompILer
```

Then, install the packages below:

```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```

## Data preparation

The propsoed datasets Split-Clothing and Split-UT-Zappos can be download from [here](https://drive.google.com/file/d/1QtD8mb6_vXrRrphgQ7JNIXcHWCwDS7VD/view?usp=sharing) and [here](https://drive.google.com/file/d/1nMLq5egLJukvGt4KOMX4jM9JfATpAY2x/view?usp=sharing) respectively.

## Instructions on running CompILer

```
bash run.sh
```

The model will be trained on 5 tasks Split-UT-Zappos, 10 tasks Split-UT-Zappos, and Split-Clothing sequentially. After each training, inference will be conducted automatically.

## Citation

**If you found our work useful for your research, please cite our work**:

```
@INPROCEEDINGS{Yanyi_2024_NeurIPS,
  author={Zhang, Yanyi and Qiu, Binglin and Jia, Qi and Liu, Yu and He, Ran},
  booktitle={Annual Conference on Neural Information Processing Systems (NeurIPS)}, 
  title={Not Just Object, But State: Compositional Incremental Learning without Forgetting}, 
  year={2024}
  }
```

Feel free to contact us: yanyi.zhang{at}mail{dot}dlut{dot}edu{dot}cn

## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

- [L2P](https://github.com/google-research/l2p)
- [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)
- [CGE](https://github.com/ExplainableML/czsl)
