# FCOS描述

## 概述

FCOS模型发表于ICCV2019，论文全名为FCOS: Fully Convolutional One-Stage Object Detection。其模型没有anchor box，通过在卷积阶段加入FPN、在计算损失阶段加入centerness这一分支，实现了更高的检测精度；通过调整模型的backbone，可以达到44.7%的AP。

如下为MindSpore使用COCO2017数据集对FCOS进行训练的示例。

## 论文

​    Zhi Tian, Chunhua Shen, Hao Chen, Tong He.FCOS: Fully Convolutional One-Stage Object Detection.[论文地址参见](https://arxiv.org/abs/1904.01355)

# 模型架构

FCOS的网络架构参见原论文图2。

backbone在本实现中采用了ResNet50+FPN，可以使用更为精密的ResNeXt-64x4d-101-FPN来达到论文中提到的44.7AP。

# 数据集

使用的数据集：[COCO2017](https://cocodataset.org/)

- 全部下载好后，文件结构为：

```
└── COCO2017
    ├── test2017                  # COCO2017的测试集，共有40,670张图片
    ├── train2017                 # COCO2017的训练集，共有118,287张图片
    ├── val2017                   # COCO2017的验证集，共有5,000张图片
    ├── captions_train2017.json
    ├── captions_val2017.json
    ├── image_info_test2017.json
    ├── image_indo_test-dev2017.json
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── person_keypoints_train2017.json
    ├── person_keypoints_val2017.json
```

- 完成下载后可尝试使用dataset/COCO_dataset.py查看是否下载并读取正确。

# 环境要求

  - 硬件
    - 准备Ascend处理器搭建硬件环境。
  - 框架
    - [MindSpore](https://www.mindspore.cn/install/en)，本模型编写时版本为r1.2，12.30更新由r1.5编写的版本。
  - 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

  通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

  - Ascend处理器环境运行


# 分布式训练

  待完善...

# 单机训练

  待完善...

# 运行评估示例

  待完善...

# 脚本说明
脚本编写中...

## 脚本及样例代码

```
└──FCOS
    ├── README.md
    ├── checkpoint                           # 训练时的checkpoint存储 
    ├── dataset                              # 数据集处理
      ├── augment.py                         # 对数据集进行随机扩充、平移等操作
      ├── COCO_dataset.py                    # 对COCO数据集进行处理
    ├── model
      ├── backbone                           # 骨干网络
        ├── resnet.py                        # 骨干网络ResNet
      ├── config.py                          # 参数配置
      ├── eval_utils.py                      # 评估时需要用到的工具
      ├── fcos.py                            # FCOS模型网络
      ├── fpn_neck.py                        # FPN处理
      ├── head.py                            # 生成模型的head
      └── loss.py                            # 生成目标框和loss损失
    ├── test_images                          # 测试图像
    ├── analyze_fail_0.dat                   # 运行报错时提及的dat文件
    ├── coco_eval.py                         # 在coco数据集上评估网络
    └── train_coco.py                        # 训练网络
```

# 脚本参数

在config.py中可以同时配置训练参数和评估参数。

```
    #backbone
    pretrained=True
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
```

# 训练过程

## 运行流程   
数据集：coco2017（仅支持解压后的coco2017）      
预训练模型：resnet50    
启动文件:train_coco.py    
数据集路径参数设置：    
1.在train_coco.py的56-57行：可以更改数据集及标签路径    
2.在resnet文件的784行：可以更改预训练模型路径    
3.在config文件中：可以更改ckpt保存路径    
运行操作过程：    
1.下载coco2017并解压后，将train_coco中的数据集及标签路径修改    
2.下载resnet50的ckpt后，将resnet文件中的预训练模型路径修改    
3.在train_coco中可以修改运行平台及运行模式    
4.运行train_coco.py文件    
（运行命令：python train_coco.py）    

eval运行说明：    
1.在coco_eval的224行：可以修改运行平台及运行模式    
2.在coco_eval的228行：可以修改数据集及标签路径    
3.在coco_eval的231行：可以修改训练好的ckpt的路径    
运行操作过程：    
1.得到训练好的ckpt    
2.修改运行说明中的参数    
3.运行coco_eval.py文件    
（运行命令：python coco_eval.py）     
 目前运行会报错如下：（正在修改）
![输入图片说明](dataset/%E5%9B%BE%E7%89%87.png)


## 用法

## Ascend处理器环境运行

待完成...

## 分布式训练

待完成...

## 单机训练

待完成...

## 分布式训练结果（8P）

待完成...

# 评估过程

## 用法

### Ascend处理器环境运行


## 评估

待完善

## 评估示例

待完善

## 结果

待完善

# 推理过程

待完善

## 在Ascend310执行推理

待完善

## 结果

待完善

# 模型描述

## 性能

### 评估性能

#### COCO2017上的FCOS

| 参数          | Ascend 910                                               |
| ------------- | -------------------------------------------------------- |
| 模型版本      |                                                          |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      |                                                          |
| MindSpore版本 | 1.2.0                                                    |
| 数据集        | COCO2017                                                 |
| 训练参数      |                                                          |
| 优化器        | SGD                                                      |
| 损失函数      | Softmax交叉熵                                            |
| 输出          | 概率                                                     |
| 损失          |                                                          |
| 速度          |                                                          |
| 总时长        |                                                          |
| 参数(M)       |                                                          |
| 微调检查点    |                                                          |
| 脚本          |                                                          |

# 随机情况说明

代码中对数据集进行了随机augmentation操作，其中含有对图片进行旋转、裁剪操作。

# ModelZoo主页

待完善