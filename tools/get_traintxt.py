# 原来文件是用来写
'''
img1_path ann1_path
img2_path ann2_path
img3_path ann3_path
img4_path ann4_path
酱紫格式的
现在 dota_dataset.py 需要
img1_path
img2_path
img3_path
img4_path
酱紫格式
'''

import os
from pathlib import Path
from os import fspath


train_list = []
train_img_root = r"..\..\DAL\DOTA_devkit\examplesplit\images" # <---- 在此处写上你的数据根目录
assert os.path.exists(train_img_root)
train_img_root = os.path.abspath(train_img_root) # 转化为绝对路径

for imgObj in Path(train_img_root).glob("**/*.png"):
    
    img_path = fspath(imgObj)
    # ann_path = img_path.replace("img_dir", "ann_dir").replace("jpg", "png")
    assert os.path.exists(img_path), f"'{img_path}' not exists"
    # assert os.path.exists(ann_path), ann_path
    
    # temp = [img_path, ann_path]
    # train_list.append(temp)
    train_list.append(img_path)

write_str = ""
train_txt = r"..\..\DAL\DOTA_devkit\examplesplit\train.txt"         # <---- 在此处写上你的txt输出目录

# for item in train_list:
#     write_str += sep.join(item) + "\n"

write_str = "\n".join(train_list)

with open(train_txt, "w") as f:
    f.write(write_str) 

print("文件写入完毕")

# 有个小bug, 就是文件的结尾多个 \n, 如果你读进来直接"split", 可能会有问题，建议读入前，先strip
