# Yolov7_Tensorrt_ROS

This project provides a detailed tutorial for how to use yolov7 in ROS based on the acceleration of tensorrt. 

Please find the Chinese user guide ./yolov7_tensorrt_ros/使用教程.pdf.

The English user guide as follows:

# Acknowledge

yolov7 https://github.com/WongKinYiu/yolov7

YOLOv7_Tensorrt https://github.com/Monday-Leo/YOLOv7_Tensorrt

# Process

```
cd yolov7_tensorrt_ros
```

create the conda environment
```
conda create -n name python=3.9
pip install -r requirements.txt
```
Use the following instructions to make sure that you have correctly install CUDA and CUDNN
```
NVCC -V
cat /user/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 
```

```
git clone https://github.com/WongKinYiu/yolov7
git clone https://github.com/Monday-Leo/YOLOv7_Tensorrt
```
