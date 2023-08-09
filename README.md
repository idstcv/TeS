# TeS
PyTorch Implementation for Our ICCV'23 Paper: "Improved Visual Fine-tuning with Natural Language Supervision"

## Requirements
* Python 3.8
* PyTorch 1.11
* transformers 4.30.2

## Usage:
TeS on CIFAR-100
* Prepare the CIFAR100 dataset.
* Download the checkpoint of ResNet50 pre-trained by MoCo from [Link](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar)
* Run the command below.
```
sh run_TeS.sh
```

## Citation
If you use the package in your research, please cite our paper:
```
@article{wang2023improved,
  title={Improved Visual Fine-tuning with Natural Language Supervision},
  author={Wang, Junyang and Xu, Yuanhong and Hu, Juhua and Yan, Ming and Sang, Jitao and Qian, Qi},
  journal={arXiv preprint arXiv:2304.01489},
  year={2023}
}
```