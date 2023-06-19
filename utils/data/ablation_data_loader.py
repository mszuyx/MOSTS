import torch
import torch.utils.data
import numpy as np
import os
import glob
import cv2

from .ablation_generate_collages import generate_collages

def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
    return -1

class ablation_data_loader(torch.utils.data.Dataset):
    def __init__(self, split='train', random_gen=None, num_candidates=5, transform_ref=None, transform=None, valid_group=0):
        self.num_candidates = num_candidates
        self.transform = transform
        self.transform_ref = transform_ref
        self.image_path = []
        self.test = []
        self.train = []
        self.split = split
        self.curriculum = 0

        if random_gen == None:
            self.random_gen = np.random.default_rng(1)
        else:
            self.random_gen = random_gen

        valid_list = [['banded','blotchy','braided','bubbly','bumpy'],
                      ['chequered','cobwebbed','cracked','crosshatched','crystalline'],
                      ['dotted','fibrous','flecked','freckled','frilly'],
                      ['gauzy','grid','grooved','honeycombed','interlaced'],
                      ['waffled', 'potholed', 'pleated', 'meshed', 'spiralled']]

        dir = '/dataset/dtd/images' # fill in your data directory here
        idx_to_class, image_path_all = self.load_path(dir)
        total_num_class = len(idx_to_class)

        valid_idx = []
        for i in range(len(valid_list[valid_group])):
            valid_idx.append(get_key(valid_list[valid_group][i],idx_to_class))
        train_idx = np.arange(total_num_class)
        train_idx = [i for i in train_idx if i not in valid_idx]

        if self.split == 'train':
            for idx in train_idx:
                self.train.append(idx_to_class[idx])
                self.image_path.append(image_path_all[idx])

        elif self.split == 'test':
            for idx in valid_idx:
                self.test.append(idx_to_class[idx])
                self.image_path.append(image_path_all[idx])

        else:
            print("wrong split option, choose from train / test")
                
        self.len = len(self.image_path)
        self.split_point = len(valid_idx)
        print("Split type: " + self.split)
        print("Total extracted classes: " + str(self.len))
        self.total_img = 0
        for i in range(self.len):
            self.total_img += len(self.image_path[i])
        print("Total # of images: "+str(self.total_img))

    def load_path(self, path):
        image_path_all = []
        classes = []
        dirs = os.listdir(path)
        dirs.sort()
        for dir in dirs:
            classes.append(dir)
            path_new = os.path.join(path, dir)
            dirs_new = sorted(glob.glob(path_new + '/*'))
            image_path_all.append(dirs_new)
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        return idx_to_class, image_path_all

    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        return image

    def __getitem__(self, index):
        index_new = index % self.len
        query_num, support_num = self.random_gen.integers(0, len(self.image_path[index_new]), size=2)
        reference_img = self.load_image(self.image_path[index_new][support_num])
        choice = np.delete(np.arange(self.len), index_new)

        num_other_class = self.random_gen.integers(1, self.num_candidates)

        if num_other_class>0:
            texture = np.zeros((num_other_class+1, 256, 256, 3))
            texture[0] = self.load_image(self.image_path[index_new][query_num])
            other_classes = self.random_gen.choice(choice, size=num_other_class, replace=True)
            for i in range(num_other_class):
                other_query_num = self.random_gen.integers(0, len(self.image_path[other_classes[i]]))
                texture[i+1] = self.load_image(self.image_path[other_classes[i]][other_query_num])
            if self.split == 'train':
                query_img, label_img = generate_collages(texture, perlin_p=0, rng=None)
            else:
                query_img, label_img = generate_collages(texture, perlin_p=0, rng=self.random_gen)
        else:
            query_img = self.load_image(self.image_path[index_new][query_num])
            label_img = np.ones((256, 256), dtype=int)

        query_img = query_img.astype(np.uint8)
        label_img = label_img.astype(np.uint8)
        reference_img = reference_img.astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=query_img, mask=label_img)
            query_img_tf = transformed["image"]
            label_img_tf = transformed["mask"]
        else:
            query_img_tf = query_img
            label_img_tf = label_img

        if self.transform_ref is not None:
            transformed_ref = self.transform_ref(image=reference_img)
            reference_img_tf = transformed_ref["image"]
        else:
            reference_img_tf = reference_img

        query_img_tf = torch.from_numpy((query_img_tf/255.0).astype(np.float32)).permute(2, 0, 1)
        label_img_tf = torch.from_numpy(label_img_tf)
        reference_img_tf = torch.from_numpy((reference_img_tf/255.0).astype(np.float32)).permute(2, 0, 1)
            
        return query_img_tf, label_img_tf, reference_img_tf, index_new+1

    def __len__(self):
        return self.total_img

