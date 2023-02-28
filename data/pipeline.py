import os
import shutil
import random
from tqdm import tqdm
import torch
import cv2
import numpy as np
from PIL import Image
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from matplotlib import pyplot as plt

def json_from_img_to_dir(root1, root2):
    # json文件转移脚本
    dirs = os.listdir(root1)
    for dir in tqdm(dirs):
        if dir[-5:] != '.json':
            continue
        json_file = os.path.join(root1, dir)
        shutil.copy(json_file, root2)
    print('abstract json_file to json_file successfully ! !')

def abstract_png_to_dir(root1, root2):
    # 提取rename后的label.png
    dirs = os.listdir(root1)
    for dir_name in tqdm(dirs):
        dir = os.path.join(root1, dir_name)
        label_name = dir_name[:-5] + '.png'
        shutil.copy(os.path.join(dir, 'label.png'), root2)
        os.rename(os.path.join(root2, 'label.png'), os.path.join(root2, label_name))
    print('abstract png to masks_dir successfully ! !')

def allocate_imgAndmask_to_maindir(root):
    # 把每一类的imgs和masks按照train和val比例，随机分配到全局的images和masks文件夹下
    types = ['close', 'slip', 'weed', 'far', 'new_far', 'new_middle']
    root_img = os.path.join(root, 'images')
    root_mask = os.path.join(root, 'masks')
    #src_images allocate
    print('src_images_allocate'.center(100,'-'))
    for type in types:
        root_type  = os.path.join(root, type)
        dir_images = os.path.join(root_type, 'images')
        dir_masks = os.path.join(root_type, 'masks')
        names_image_json = os.listdir(dir_images)
        length  = len(names_image_json)//2
        num_train = int(0.85*length)
        train_idxs= random.sample(list(range(length)), num_train)
        count = 0
        for name_image_json in tqdm(names_image_json):
            if name_image_json[-4:] != '.jpg':
                continue
            image = os.path.join(dir_images, name_image_json)
            mask_name = name_image_json.split('.')[0] + '.png'
            mask = os.path.join(dir_masks, mask_name)
            if count in train_idxs:
                shutil.copy(image, os.path.join(root_img, 'train'))
                shutil.copy(mask, root_mask)
            else:
                shutil.copy(image, os.path.join(root_img, 'val'))
                shutil.copy(mask, root_mask)
            count += 1
        print(f'src_{type} has been allocated successfully! !')

    #aug_images allocate
    print('aug_images_allocate'.center(100,'-'))
    for type in types:
        root_type  = os.path.join(root, type)
        dir_images = os.path.join(root_type, 'images_aug')
        names_image = os.listdir(dir_images)
        length  = len(names_image)
        num_train = int(0.85*length)
        train_idxs= random.sample(list(range(length)), num_train)
        count = 0
        for name_image in tqdm(names_image):
            image = os.path.join(dir_images, name_image)
            if count in train_idxs:
                shutil.copy(image, os.path.join(root_img, 'train'))
            else:
                shutil.copy(image, os.path.join(root_img, 'val'))
            count += 1
        print(f'aug_{type} has been allocated successfully! !\n\n')
    print('allocate train_val datasets successfully! !')
    
        

    pass

def inspect_loss_or_duli(root1, root2):
    # root1 = '/media/rjg/SSD/crop_line/images'
    # root2 = '/media/rjg/SSD/crop_line/masks' 
    img_list = []
    mask_list = []
    error_list = []
    for type in ['train', 'val']:
          dirs = os.listdir(os.path.join(root1, type))
          for dir in tqdm(dirs):
            if dir[-4:] != '.jpg':
                continue
            if dir[:-4] in img_list:
                print(dir[:-4])
            else:
                img_list.append(dir[:-4])
    
    dirs = os.listdir(root2)
    for dir in dirs:
        mask_list.append(dir[:-4])
    
    for img in img_list:
        if img not in mask_list:
            error_list.append(img)
            print(img)
    print('total num: ', len(error_list))
    print(len(img_list))
    print(len(mask_list))

def augment(img_path, mask_path, type):
    #高斯噪声和模糊
    train_transform0 = Compose([
            OneOf([
            transforms.GaussNoise(var_limit=(1000,3000), mean=5,p=1),
            #transforms.GaussNoise(var_limit=(100,300), mean=5,p=1),
            transforms.GaussianBlur(blur_limit=(9,15), sigma_limit=15, p=1),
            transforms.GlassBlur(p=1),
            ], p=1)
    ])
    #亮度和对比度
    train_transform1 = Compose([
            OneOf([
            transforms.HueSaturationValue(p=1),
            transforms.RandomBrightnessContrast(0.3,0.16,p=1),
            ], p=1),
    ])
    #阴影遮挡
    train_transform2 = Compose([
            transforms.RandomShadow(shadow_roi=(0, 0.5, 1, 1), 
                                num_shadows_lower=1,
                                num_shadows_upper=3,
                                shadow_dimension=6, p=1),
    ])
    #混合
    train_transform3 = Compose([
        transforms.RandomShadow(shadow_roi=(0, 0.5, 1, 1), 
                                num_shadows_lower=1,
                                num_shadows_upper=3,
                                shadow_dimension=6, p=0.5),
        OneOf([
            transforms.GaussNoise(var_limit=(1000,3000), mean=5,p=0.5),
            #transforms.GaussNoise(var_limit=(100,300), mean=5,p=1),
            transforms.GaussianBlur(blur_limit=(9,15), sigma_limit=15, p=0.5),
            transforms.GlassBlur(p=0.5),
            ], p=0.5),
        OneOf([
            transforms.HueSaturationValue(p=0.5),
            transforms.RandomBrightnessContrast(0.3,0.16,p=0.5),
            ], p=0.5),
        OneOf([
            transforms.RandomRain(p=0.5),
            transforms.RandomSnow(p=0.2),
            transforms.RandomSunFlare(p=0.2),
        ], p=0.2),
    ])

    table_augments = {0:train_transform0,
                      1:train_transform1,
                      2:train_transform2,
                      3:train_transform3,}
    
    img, mask = np.array(Image.open(img_path)), np.array(Image.open(mask_path))
    augmented = table_augments[type](image = img, mask = mask)
    img_aug, mask_aug = Image.fromarray(augmented['image']), Image.fromarray(augmented['mask'], "P")
    return img_aug, mask_aug

def path_augment(root, type_list=[0,1,2,3]):
    palette_data = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
                0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0]  # 调色板
    for img_name in tqdm(os.listdir(os.path.join(root, 'images'))):
        if img_name[-4:] != '.jpg':
            continue
        base_name = img_name[:-4]
        mask_name = base_name + '.png'
        img = os.path.join(root, 'images', img_name)
        mask = os.path.join(root, 'masks', mask_name)
        for type in type_list:
            new_image_name = base_name + '_aug%d'%type + '.jpg'
            # new_mask_name = base_name + '_aug%d'%type + '.png'
            # img_aug, mask_aug = augment(img, mask, type) #PIL Image
            #mask_aug.putpalette(palette_data)#调色板
            img_aug, _ = augment(img, mask, type)
            img_aug.save(os.path.join(root, 'images_aug', new_image_name))
            #mask_aug.save(os.path.join(root, 'masks_aug', new_mask_name))
    pass

if __name__ == '__main__':
    #1.给每一个类别创建一个文件夹，然后里面是images、masks、json_file、json_to_dataset。
    #2.在image中标注，json文件就存放在image中
    #3.将json文件批量复制到../json_file中，即运行 json_from_img_to_dir()
    #3.labelme批量转化，labelme_json_to_dataset ./json_file --out ./json_to_dataset
    #4.将json_to_dataset中的png图片复制到../masks中，即运行 abstract_png_to_dir()
    #5.检查是否有问题数据，即运行 inspect_loss_or_duli()
    #6.数据增强,填补images和masks中的图片数量，即运行 path_augment()

    #7.对每一类都运行步骤1~5(6)，最后将它们主文件下的images(带着json文件)随机分配到./images的train和val中。
    #  并将mask都移动到./masks下，即运行 allocate_imgAndmask_to_maindir()

    root = '/home/rjg/dataset2'

    root1 = '/home/rjg/dataset2/weed/json_to_dataset'
    root2 = '/home/rjg/dataset2/weed/masks'

    
    allocate_imgAndmask_to_maindir(root)


    # kind_list = ['close', 'slip', 'weed', 'far', 'new_far', 'new_middle']
    # for kind in kind_list:
    #     root1 = os.path.join(root, kind)
    #     path_augment(root1)
    #     print(f'{kind} augment finished!!')
    # print('Augment finished successfully ! !')

    #allocate_imgAndmask_to_maindir(root)


    #经验：
    #1.判断json文件可不可以转换为json_to_dataset时，复制一堆json文件到一个文件夹下，然后执行一次labelme_json_to_dataset，删除第一个文件
    # （只要终端卡住，就说明这个json文件是valid的，直接中止转换即可。如果json文件有问题，会直接报错）
    #
    #2.对于mask文件，也需要人工检查一下，有的是否能显示出来
    #
    #3.对于图像增强后，也需要检查一下图片是否正常，是否丢失
    #
    #4.有时候原图images也可能出现突然丢失的情况，但是你总览时看不出来，缩放图还是正常的。最好的办法是把整个文件目录拷贝到本地，有问题的图片就一目了然了！！