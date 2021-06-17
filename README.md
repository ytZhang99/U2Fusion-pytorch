# U2Fusion: A Unified Unsupervised Image Fusion Network
- This is the PyTorch implementation of [U2Fusion: A Unified Unsupervised Image Fusion Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9151265) (TPAMI 2020).
- The original paper can solve 3 typical image fusion tasks(multi-modal image fusion, multi-exposure image fusion and multi-focus image fusion), but in this repository, **only the multi-exposure image fusion branch** was implemented.

![framework](./images/framework.png)

## 1. Environment
- Python >= 3.7
- PyTorch >= 1.4.0 is recommended
- opencv-python = 4.5.1
- matplotlib
- tensorboard
- pytorch_msssim

## 2. Dataset
The training data and testing data is from the [[SICE dataset]](https://github.com/csjcai/SICE, "Official SICE").

## 3. Test
1. Clone this repository:
    ```
    To be updated
    ```
2. Place the over-exposed images and under-exposed images in `dataset/test_data/over` and `dataset/test_data/under`, respectively.
3. Run the following command for multi-exposure fusion:
    ```
    python main.py --test_only
    ```
4. Finally, you can find the Super-resolved and Fused results in `./test_results`.

## 4. Training
To be updated

## 5. Citation
The following paper might be cited:
```
@article{xu2020u2fusion,
  title={U2Fusion: A unified unsupervised image fusion network},
  author={Xu, Han and Ma, Jiayi and Jiang, Junjun and Guo, Xiaojie and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```
If you find this repository helpful in your research or publication, you may cite:

