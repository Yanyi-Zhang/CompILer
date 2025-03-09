# Pytorch Implementation for Compositional Incremental Learner (CompILer)

## Environment preparation

we follow the [l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch) setup environment instructions.

1. Create python environment using conda.

```bash
conda create -n compiler python=3.10
conda activate compiler
```

2. Install dependencies.

```bash
# install pytorch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# other dependencies
pip install -r requirements.txt
```

## Dataset preparation

**Split-Clothing**

Download and unzip`images.zip` from [Google drive](https://drive.google.com/drive/folders/1ky5BvTFrMkPBdAWixHFGLdcfJHfu5e9_) to `local_datasets/clothing16k`

**Split-UT-Zappos**

Download the dataset by:

```bash
bash download_ut.sh
```

We also showcase some images as samples in the `data_sample/`.

## Training and testing

To train a model via command line:

```bash
bash run.sh
```

The model will be trained on 5 tasks Split-UT-Zappos, 10 tasks Split-UT-Zappos, and Split-Clothing sequentially. After each training, inference will be conducted automatically."