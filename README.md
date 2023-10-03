# Demo Code for "Talking Head Anime from a Single Image"


  
该仓库含有两种应用：  
  
* *manual poser* 允许用户通过**拖动滑块**改变二次元角色的姿态
* *puppeteer* 允许面捕控制

## Try the Manual Poser on Google Colab

可以在Colab中尝试[该项目](https://colab.research.google.com/github/pkhungurn/talking-head-anime-demo/blob/master/tha_colab.ipynb)

## 硬件要求

需要一个强力的Nvidia GPU来运行该项目。比如GTX 1080Ti或者 Titan RTX 

其实`manual poser`模式下，也可以使用 CPU


## 环境依赖

* Python >= 3.6
* pytorch >= 1.4.0
* dlib >= 19.19
* opencv-python >= 4.1.0.30
* pillow >= 7.0.0
* numpy >= 1.17.l2



## 创建虚拟环境

```bash
创建
python -m venv [文件路径]

激活
C:\> <venv>\Scripts\activate.bat
```
> 不创建也行(= =) 直接全局梭哈

## 准备数据
 
下载模型权重文件：

链接：https://pan.baidu.com/s/1VmLncRBTl4zJMo4nHsUf4w  
密码：jack

主要是data目录下的3个.pt文件和1个.bat文件

目录结构应该如下所示：

```
+ data
  + illust
    - placeholder.txt
    - waifu_00_256.png
    - waifu_01_256.png
    - waifu_02_256.png
    - waifu_03_256.png
    - waifu_04_256.png
  - combiner.pt
  - face_morpher.pt
  - placeholder.txt
  - shape_predictor_68_face_landmarks.dat
  - two_algo_face_rotator.pt
```

可以使用`data/illust`中提供的5张图片. 也可以自己准备.  
 图片需要满足以下要求：
* PNG格式  
* 大小256 x 256.
* 角色头部包含在128x128的框中
* 4通道RGBA
* 透明背景

 图片应保存在`data/illust`文件夹中

## Running the Program

调整当前目录为项目根目录，然后运行下面的命令：

滑块拖动

> `python app/manual_poser.py`

实时面捕（没有GPU就别试了）

> `python app/puppeteer.py`

## 其它

使用 [a face tracker code implemented by KwanHua Lee](https://github.com/lincolnhard/head-pose-estimation) 来实现**面捕**

