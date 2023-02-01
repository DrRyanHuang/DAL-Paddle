python -m paddle.distributed.launch train.py  \
--dataset  VOC  \
--train_path  /home/aistudio/data/data21544/PascalVOC2007/VOC2007_train_val/ImageSets/Main/trainval.txt  \
--test_path  /home/aistudio/data/data21544/PascalVOC2007/VOC2007_train_val/ImageSets/Main/val.txt  \
--resume  \
--fleet