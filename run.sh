bash tools/dist_train.sh projects/VGGTDet/config/vggtdet_scannet.py 1

bash tools/dist_test.sh projects/VGGTDet/config/vggtdet_scannet.py /mnt/workspace/pretrain/VGGT-Det/epoch_180.pth 1
