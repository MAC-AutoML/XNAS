#!/usr/bin/env bash
if [ "$1" == "imagenet" ]; then
mount -t tmpfs -o size=160G tmpfs /userhome/temp_data
mkdir /userhome/temp_data/ImageNet
mkdir /userhome/temp_data/ImageNet/train
tar xvf /gdata/ImageNet2012/ILSVRC2012_img_train.tar -C /userhome/temp_data/ImageNet/train
cp /userhome/script/unzip_image_net.sh /userhome/temp_data/ImageNet/train/
cd /userhome/temp_data/ImageNet/train/
./unzip_image_net.sh
cp -r /userhome/ILSVRC2012_img_val/ /userhome/temp_data/ImageNet/val
elif [ "$1" == 'cifar10' ];
then
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /userhome/data/cifar10 /userhome/temp_data/
elif [ "$1" == 'cifar100' ];
then
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /userhome/data/cifar100 /userhome/temp_data/
elif [ "$1" == 'fashionmnist' ];
then
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /userhome/data/fashionmnist /userhome/temp_data/
fi