# Generating Editable Head Avatars with 3D Gaussian GANs (EGG3D)

[[project page](https://liguohao96.github.io/EGG3D)]
[[arXiv Paper](https://arxiv.org/abs/2412.19149)]

EGG3D generate animatable and editable 3D head with 3D-aware GANs and 3DGS.

## Change Logs

- 2024/12/31: initial commit

## Setups

- **clone**

    ```git clone https://github.com/liguohao96/EGG3D.git --recursive```

- **environment**

    install PyTorch, we recomand install PyTorch 1.13.1 with CUDA 11.7

    install other python packages through
    `pip install -r requirements.txt`.

- **face model**

    **FLAME**

    please first down `FLAME 2020` and `FLAME texture space` from [here](https://flame.is.tue.mpg.de/download.php) and put extracted files under `Data/FLAME2020`, the directory should be like: 
    ```
    Data/FLAME2020
    ├── female_model.pkl
    ├── FLAME_texture.npz
    ├── generic_model.pkl
    ├── male_model.pkl
    ├── tex_mean.png 
    └── ...
    ```

- **fitted condition params**

    Down fitted condition params of FFHQ@512 from [here](https://drive.google.com/file/d/1WKMRA-knjWexGiuNBQlFNrZfnXOO6q4d/view?usp=drive_link) and put under `Data/fitted_dataset`.

## Checkpoint

|Training Dataset|Link|
|:-:|:-:|
|FFHQ@512|[google drive](https://drive.google.com/file/d/1J0ckbmvaDSgowD4Tq7uNmBZOW3fygMDz/view?usp=drive_link)|

## Inference

First, download `fitted condition params` and `checkpoint`.
Then, generate images through
```shell
python scripts/application/noise_to_3d.py --checkpoint GAN_3dgs_512.zip
```
It will save intermediate result and final mesh at `./temp/GAN_3dgs_512/noise_to_3d`

## Acknowledgments

Our network codes are based on [EG3D](https://github.com/NVlabs/eg3d), thanks for their great works.


## Citation

```
@inproceedings{li2025egg3d,
  title  = {Generating Editable Head Avatars with 3D Gaussian GANs}, 
  author = {Guohao Li and Hongyu Yang and Yifang Men and Di Huang and Weixin Li and Ruijie Yang and Yunhong Wang},
  booktitle = {ICASSP 2025},
  year   = {2025},
}
```
