# Yolov5基线部署过程文档

## 基础环境
Python>=3.8
PyTorch>=1.6 

安装依赖：pip install -r requirements.txt

## 数据集准备
### 目录格式
在master目录下新建VOCData文件夹，放入训练数据

|VOCData 

|--images # 存放图片

|--Annotations # 存放图片对应的xml文件

### 划分数据集

新建split_train_val.py，实现数据集划分，生成ImageSets文件夹。划分后文件夹目录格式

|VOCData 

|--images # 存放图片

|--Annotations # 存放图片对应的xml文件

|--ImageSets/Main #之后会在Main文件夹内自动生成train.txt，val.txt，test.txt和trainval.txt四个文件，存放训练集、验证集、测试集图片的名字（无后缀.jpg）

此时默认无测试集划分

默认划分：0.9训练+0.1验证

### 标记格式转换：XML转yolo

在VOCData目录下创建text_to_yolo.py并运行，实现格式转换。（将文件中master_path修改为自己master的路径）

运行后会生成labels文件夹和dataSet_path文件夹。

* 其中labels中为不同图像的标注文件。每个图像对应一个txt文件，文件每一行为一个目标的信息，包括class, x_center, y_center, width, height格式，即为 yolo_txt格式

* dataSet_path文件夹包含三个数据集的txt文件，train.txt等txt文件为划分后图像所在位置的绝对路径，如train.txt就含有所有训练集图像的绝对路径

### 配置yaml文件
在 yolov5 目录下的 data 文件夹下 新建一个 myvoc.yaml文件（可以自定义命名），用记事本打开。

内容是：训练集以及验证集（train.txt和val.txt）绝对路径（通过 text_to_yolo.py 生成），然后是目标的类别数目和类别名称。

## k-means聚类生成先验框
### 生成anchors
在VOCData目录下创建程序两个程序 kmeans.py 以及 clauculate_anchors.py，修改路径、类别标签，运行 clauculate_anchors.py，生成anchors.txt文件。

### 配置模型文件
选择一个版本的yolov5模型，在yolov5目录下的model文件夹下是模型的配置文件，有n、s、m、l、x版本，逐渐增大（随着架构的增大，训练时间也是逐渐增大）。

这里选用yolov5s.yaml。修改类别数为2，将anchors.txt里的best anchors对应填入yolov5s.yaml的anchors中，四舍五入取整。

## 训练模型与可视化
### 参数意义

weights：权重文件路径

cfg：存储模型结构的配置文件

data：存储训练、测试数据的文件

epochs：指的就是训练过程中整个数据集将被迭代多少次，显卡不行你就调小点。

batch-size：一次看完多少张图片才进行权重更新，梯度下降的mini-batch,显卡不行你就调小点。

img-size：输入图片宽高，显卡不行你就调小点。

device：cuda device, i.e. 0 or 0,1,2,3 or cpu。选择使用GPU还是CPU

### 训练命令
cd到yolov5_master根目录，运行

python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/myvoc.yaml --epoch 200 --batch-size 8 --img 640   --device cpu

–weights weights/yolov5s.pt ：yolov5的pt文件存放路径（新建一个weights文件夹）

–epoch 200 ：训练200次

–batch-size 8：训练8张图片后进行权重更新

–device cpu：使用CPU训练

### 可视化
训练时或者训练后可以利用 tensorboard 查看训练可视化

tensorboard --logdir=runs

## 模型测试
使用刚刚训练出的最好的模型 best.pt 来测试，在yolov5目录下的 runs/train/exp/weights

python detect.py --weights runs/train/exp/weights/best.pt --source ../data/video/tram.mp4

## 快速部署
复制打包好图片的完整版yolov5_master

安装依赖：pip install -r requirements.txt

VOCData目录下text_to_yolo.py、clauculate_anchors.py代码中的yolov5_master绝对路径修改

分别运行text_to_yolo.py、clauculate_anchors.py

data目录下myvoc.yaml绝对路径修改

cd到yolov5_master根目录，运行
python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/myvoc.yaml --epoch 200 --batch-size 8 --img 640   --device cpu
