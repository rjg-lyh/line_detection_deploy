import os
import torch.utils.data as data
import numpy as np

from PIL import Image

def table_cmp(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    table = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        table[i] = np.array([r, g, b])

    table = table/255 if normalized else table
    return table

class Crop_line(data.Dataset):

    table = table_cmp()
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):
        
        self.images,  self.masks = self.split_dataset(root, image_set)
        self.transform = transform
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    def split_dataset(self, root, image_set):
        path_img = os.path.join(root, 'images', image_set)
        path_mask = os.path.join(root, 'masks')
        imgs = os.listdir(path_img)
        images, masks = [], []
        for img in imgs:
            # if img[-4:] != '.jpg':
            #     continue
            img_name = os.path.basename(img)
            if img_name.split('.')[0][-4:-1] == 'aug':
                mask_name = img_name.split('.')[0][:-5] + '.png'
            else:
                mask_name = img_name.split('.')[0] + '.png'
            img = os.path.join(path_img, img_name)
            # mask_pkgname = img_name.split('.')[0] + '_json'
            # mask_pkg = os.path.join(path_mask, mask_pkgname)
            #mask = os.path.join(mask_pkg, 'label.png')
            mask = os.path.join(path_mask, mask_name)
            images.append(img)
            masks.append(mask)
        return images, masks
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.table[mask]

if __name__ == '__main__':
    import shutil
    import os
    from tqdm import tqdm

    # root = '/home/rjg/project/data/crop_line/images'
    # root1 = '/home/rjg/project/data/crop_line/json_to_datasets'
    # #json文件转移脚本
    # for type in ['train', 'val']:
    #       dirs = os.listdir(os.path.join(root, type))
    #       for dir in tqdm(dirs):
    #         if dir[-5:] != '.json':
    #             continue
    #         json_file = os.path.join(root, type, dir)
    #         shutil.copy(json_file, root1)
    # #最后终端运行 labelme_json_to_dataset json_files --out json_to_datasets

    # root = '/media/rjg/SSD/crop_line/json_to_datasets' 
    # root1 = '/media/rjg/SSD/crop_line/masks'
    
    # dirs = os.listdir(root)
    # for dir_name in tqdm(dirs):
    #     dir = os.path.join(root, dir_name)
    #     label_name = dir_name[:-5] + '.png'
    #     os.rename(os.path.join(dir, 'label.png'), os.path.join(dir, label_name))
    #     shutil.copy(os.path.join(dir, label_name), root1)

    # root = '/media/rjg/SSD/crop_line/images'
    # root1 = '/media/rjg/SSD/crop_line/masks'
    
    # img_list = []
    # mask_list = []
    # error_list = []
    # for type in ['train', 'val']:
    #       dirs = os.listdir(os.path.join(root, type))
    #       for dir in tqdm(dirs):
    #         if dir[-4:] != '.jpg':
    #             continue
    #         if dir[:-4] in img_list:
    #             print(dir[:-4])
    #         else:
    #             img_list.append(dir[:-4])
    
    # dirs = os.listdir(root1)
    # for dir in dirs:
    #     mask_list.append(dir[:-4])
    
    # for img in img_list:
    #     if img not in mask_list:
    #         error_list.append(img)
    #         print(img)
    # print('total num: ', len(error_list))
    # print(len(img_list))
    # print(len(mask_list))
    
        
    