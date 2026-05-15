## Environment Set Up
Install required packages:
```
conda create -n cpgg python=3.9
conda activate cpgg
pip install -r requirements.txt // 不够，看下边
```

## Train Models
```
sh script_train_phenotype_vae.sh
sh script_vae3d_kl8.sh
sh script_train_cpgg.sh
```
## Generate CMR
```
sh script_sample.sh
```
## 我的补充
现在 requirements.txt 直接安装还不够，主要剩这些：

PyTorch CUDA wheel 源

建议先单独装 torch/torchvision：
```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

LPIPS 权重文件 vgg.pth

这个不能可靠地靠 requirements 安装，因为服务器访问下载源会卡。需要手动放到项目目录：

CPGG/taming/modules/autoencoder/lpips/vgg.pth
验证：
```
python -c "import torch; p='taming/modules/autoencoder/lpips/vgg.pth'; ckpt=torch.load(p, map_location='cpu'); print(type(ckpt)); print(list(ckpt.keys())[:5])"
```
训练时环境变量

跑 3D VAE 建议带上：

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
