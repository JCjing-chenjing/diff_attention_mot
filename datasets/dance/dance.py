import os
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch.utils.data import DataLoader
#from torchvision import  transforms
from collections import defaultdict
from PIL import Image, ImageDraw
import datasets.dance.transforms as T
# import transforms as T

class DanceDataset(Dataset):
    def __init__(self, args, image_set="train", transform=None):
        self.args = args
        self.img_size = args.img_size
        if transform is None:
            transform = self.get_transforms(image_set)
        self.transform = transform
        self.root_dir = os.path.join(args.source, image_set)
        self.all_labels = defaultdict(lambda : defaultdict( lambda: defaultdict(list)))
        # self.all_labels = defaultdict(lambda : defaultdict(list))
        self.indices = []
        self.read_all_labels()
        
        # if args.det_db:
        #     with open(os.path.join(args.mot_path, args.det_db)) as f:
        #         self.det_db = json.load(f)
        # else:
        #     self.det_db = defaultdict(list)

    def read_all_labels(self):
        for vid in os.listdir(self.root_dir):
            if 'seqmap' == vid:
                continue
            # vid = os.path.join(split_dir, vid)
            # if 'DPM' in vid or 'FRCNN' in vid:
            #     print(f'filter {vid}')
            #     continue
            gt_path = os.path.join(self.root_dir, vid, 'gt', 'gt.txt')
            pre_frame = None
            for l in open(gt_path):
                t, i, *xywh, mark, label = l.strip().split(',')[:8]
                t, i, mark, label = map(int, (t, i, mark, label))
                if mark == 0:
                    continue
                if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                    continue
                # else:
                #     crowd = False
                x, y, w, h = map(float, (xywh))
                self.all_labels[vid][t][i].append([x, y, x+w, y+h])
                # self.all_labels[vid][t] = [x, y, w, h, i]
                if pre_frame:
                    self.indices.append((pre_frame, (vid, t)))
                pre_frame = (vid, t)
  


    def __len__(self):
        return len(self.indices)       
    
    def __getitem__(self, idx):
        (pre_vid, pre_f_index), (cur_vid, cur_f_index)= self.indices[idx]

        pre_images, pre_targets = self.load_image(pre_vid, pre_f_index)
        cur_images, cur_targets = self.load_image(cur_vid, cur_f_index, pre_targets["obj_ids"])

        if self.transform is not None:
            pre_images, pre_targets = self.transform(pre_images, pre_targets)
            cur_images, cur_targets = self.transform(cur_images, cur_targets)
 
        return {
            'pre_images': pre_images, 
            "cur_images": cur_images,
            'pre_targets': pre_targets,
            'cur_targets': cur_targets,
        }

        # return pre_images, cur_images, pre_targets, cur_targets

    def load_image(self, vid, idx: int, pre_obj_ids=None):
        img_path = os.path.join(self.root_dir, vid, 'img1', f'{idx:08d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img.size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        # obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.

        targets['boxes'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        # targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        if pre_obj_ids is None:
            for id in self.all_labels[vid][idx]:
                for xywh in self.all_labels[vid][idx][id]:
                    targets['boxes'].append(xywh)
                    targets['obj_ids'].append(id)
                    targets['scores'].append(1.)
        else:
            for id in pre_obj_ids.numpy():
                if  id in  self.all_labels[vid][idx]:
                    for xywh in self.all_labels[vid][idx][id]:
                        targets['boxes'].append(xywh)
                        targets['obj_ids'].append(id)
                        targets['scores'].append(1.)
                else:
                    targets['boxes'].append([0,0,0,0])
                    targets['obj_ids'].append(id)
                    targets['scores'].append(0.0)

        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float32)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        # targets['boxes'][:, 2:] += targets['boxes'][:, :2]
        targets['img_path']=img_path
        targets['boxes_ori'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)

        return img, targets

    
    def get_transforms(self, image_set):

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
        scales = [540,630,720]
        if image_set == 'train':
            return T.Compose([
                # T.MotRandomHorizontalFlip(),
                # T.MotRandomSelect(
                #     T.MotRandomResize(scales, max_size=1536),
                #     T.MotCompose([
                #         # T.MotRandomResize([800, 1000, 1200]),
                #         T.MotRandomResize([960]),
                #         T.FixedMotRandomCrop(800, 1200),
                #         T.MotRandomResize(scales, max_size=1536),
                #     ])
                # ),
                # T.MOTHSV(),
                # T.RandomResize([(360, 640)]),
                T.RandomResize(self.img_size),
                normalize,
            ])

        if image_set == 'val':
            return T.Compose([
                # T.MotRandomResize([800], max_size=1333),
                # T.RandomResize([(360, 640)]),
                T.RandomResize(self.img_size),
                
                normalize,
            ])

        raise ValueError(f'unknown {image_set}')


from typing import Optional, List
def mot_collate_fn(batch: List[dict]) -> dict:
    ret_dict = {}
    for key in list(batch[0].keys()):
        # assert not isinstance(batch[0][key], torch.Tensor)
        ret_dict[key] = [img_info[key] for img_info in batch]
        if len(ret_dict[key]) == 1:
            ret_dict[key] = ret_dict[key][0]
    return ret_dict

def collate_fn(batch):
    ret_dict = {}
    for key in list(batch[0].keys()):
        # assert not isinstance(batch[0][key], torch.Tensor)
        ret_dict[key] = [img_info[key] for img_info in batch]
        if len(ret_dict[key]) == 1:                                             #batch_size = 1
            if isinstance(ret_dict[key][0], torch.Tensor):
                ret_dict[key] = ret_dict[key][0].unsqueeze(0)      
            elif isinstance(ret_dict[key][0], dict):
                sub_dict = {}
                for sub_k in ret_dict[key][0].keys():
                    sub_dict[sub_k] = ret_dict[key][0][sub_k].unsqueeze(0)  if isinstance(ret_dict[key][0][sub_k], torch.Tensor) else ret_dict[key][0][sub_k]
                ret_dict[key] = sub_dict
            else:
                ret_dict[key] = ret_dict[key][0]
        else:                                                                   #batch_size > 1
            if isinstance(ret_dict[key][0], dict):
                sub_dict = {}
                for sub_k in  ret_dict[key][0].keys(): 
                    sub_dict[sub_k] = [sub_info[sub_k] for sub_info in ret_dict[key]]
                
                for sub_k in sub_dict:
                    max_shape = [(data.shape) for data in sub_dict[sub_k]]
                    max_shape = max(max_shape)
                    if not len(max_shape):
                        continue
                    tmp_data_list = []
                    for data in sub_dict[sub_k]:
                        tmp_data = torch.zeros(max_shape)
                        n = data.shape
                        tmp_data[:n[0]] = data
                        tmp_data_list.append(tmp_data)
                    sub_dict[sub_k] = torch.stack(tmp_data_list, 0)
                ret_dict[key] = sub_dict
            else:
                ret_dict[key] = torch.stack(ret_dict[key], 0)

        
    return ret_dict


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = r"E:\work\code\motrv2\data\DanceTrack"
    data_set = DanceDataset(data_dir, image_set="train")
    test_loader = DataLoader(dataset=data_set, batch_size=2, collate_fn=mot_collate_fn, shuffle=True, num_workers=0, drop_last=False)
    # test_loader = DataLoader(dataset=data_set, batch_size=2, shuffle=True, num_workers=0, drop_last=False)

    # for i in range(1):
    #     print(data[i])
    for data in test_loader:
        pre_imgs = data['pre_imgs']
        cur_images = data['cur_images']
        pre_target = data['pre_targets']
        cur_target = data['cur_targets']
        print(pre_imgs[0].shape, cur_images[1].shape,) # pre_target, cur_target)
