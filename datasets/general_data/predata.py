from PIL import Image
import os
import torch
import torchvision.transforms as T
from pathlib import Path
import numpy as np


class PreData():


    def __init__(self,
                 data_dir,
                 type='dance',
                 img_size=(720, 540),
                 gt_postfix='gt/gt.txt',
                 img_postfix='img1',
                 img_format='jpg'
                 ):


        self.data_dir = data_dir
        self.type = type
        self.gt_postfix = gt_postfix
        self.img_postfix = img_postfix
        self.img_size = img_size
        self.img_format = img_format
        self.file_name_lst = self.get_file_name_lst()
        self.seq_info = self.get_seq_info()
        self.transfomer=self.trans()

    def get_file_name_lst(self):

        file_name_lst=os.listdir(self.data_dir)

        file_name_lst=[file_name for file_name in file_name_lst if not os.path.isfile(os.path.join(self.data_dir,file_name))]
        if len(file_name_lst)<0: raise FileExistsError("lacking file in get_file_name_lst")
        return file_name_lst


    def get_seq_info(self):
        seq_info={}
        for file_name in self.file_name_lst:

            img_path=os.path.join(self.data_dir,file_name,self.img_postfix)
            gt_txt_path=os.path.join(self.data_dir,file_name,self.gt_postfix)
            img_name_lst=[img_name for img_name in os.listdir(img_path) if img_name[-3:]==self.img_format]
            if len(img_name_lst)>1:
                img_path_lst = [ os.path.join(img_path,img_name) for img_name in img_name_lst]
                frame_id=np.array([int(img_name[:-4]) for img_name in img_name_lst])
                index=np.argsort(frame_id)
                seqLength=len(frame_id)
                seq_info[file_name]={
                    'img_dir': img_path,  # 当前文件夹主路径
                    'img_path':img_path_lst,  # 当前文件夹图像绝对路径
                     'img_names':img_name_lst,  # 当前文件夹图像名称
                     'gt_txt_path':gt_txt_path,  # gt.txt文件绝对路径
                     'frame_id':frame_id,  # 以图像名称名作为frame_id，从1开始
                     'index':index,  # 按照sort升序得到的索引
                     'seqLength':seqLength  # 测试图像总数量
                                     }

        return seq_info



    def parse_gt(self,info):
        '''
        return-->gt_info:必须按照trackeval中的gt格式，后续才不会出错
        '''
        gt_path=info['gt_txt_path']
        if self.type=='dance':

            gt_info=self.parse_dance_txt(gt_path)

        return gt_info

    def parse_dance_txt(self,gt_path):
        gt_info = self.read_txt(gt_path)

        gt_info = np.array([[float(v) for v in det.split(',')] for det in gt_info])

        return gt_info



    def trans(self):
        normalize = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    def parse_data(self,info,id, gt_info=None):
        data_dict = {}
        frame_id = info['frame_id'][id]
        frame_idx = gt_info[:, 0] == frame_id

        boxes_xywh = gt_info[frame_idx, 2:6]
        track_id = gt_info[frame_idx, 1]
        img_name = info['img_names'][id]
        img_path=info['img_path'][id]
        img_ori = Image.open(img_path)

        img=self.transfomer(img_ori)

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(self.img_size,img_ori.size ))
        ratio_width, ratio_height = ratios
        resize_w,resize_h=self.img_size

        resize_boxes = torch.as_tensor(boxes_xywh) * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        resize_boxes = resize_boxes / torch.tensor([resize_w, resize_h, resize_w, resize_h], dtype=torch.float32)
        data_dict['frame_id'] = frame_id
        data_dict['track_id'] = track_id
        data_dict['boxes_xywh_ori'] = boxes_xywh
        data_dict['img_path'] = img_path
        data_dict['img'] = img
        data_dict['boxes_xywh'] = resize_boxes
        data_dict['img_name'] = img_name



        return data_dict



    def process_data(self,info,gt_info,idx):


        cur_id=info['index'][idx]
        cur_data_dict=self.parse_data(info,cur_id, gt_info)
        pre_data_dict = {}
        if idx>0:
            pre_id=info['index'][idx-1]
            pre_data_dict = self.parse_data(info, pre_id, gt_info)
        return pre_data_dict,cur_data_dict













    # def read_all_labels(self):
    #     for vid in os.listdir(self.root_dir):
    #         if 'seqmap' == vid:
    #             continue
    #         # vid = os.path.join(split_dir, vid)
    #         # if 'DPM' in vid or 'FRCNN' in vid:
    #         #     print(f'filter {vid}')
    #         #     continue
    #         gt_path = os.path.join(self.root_dir, vid, 'gt', 'gt.txt')
    #         pre_frame = None
    #         for l in open(gt_path):
    #             t, i, *xywh, mark, label = l.strip().split(',')[:8]
    #             t, i, mark, label = map(int, (t, i, mark, label))
    #             if mark == 0:
    #                 continue
    #             if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
    #                 continue
    #             # else:
    #             #     crowd = False
    #             x, y, w, h = map(float, (xywh))
    #             self.all_labels[vid][t][i].append([x, y, x + w, y + h])
    #             # self.all_labels[vid][t] = [x, y, w, h, i]
    #             if pre_frame:
    #                 self.indices.append((pre_frame, (vid, t)))
    #             pre_frame = (vid, t)
    #
    #
    # def load_image( vid, idx: int, pre_obj_ids=None):
    #     img_path = os.path.join(self.root_dir, vid, 'img1', f'{idx:08d}.jpg')
    #     img = Image.open(img_path)
    #     targets = {}
    #     w, h = img._size
    #     assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
    #     # obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.
    #
    #     targets['boxes'] = []
    #     targets['obj_ids'] = []
    #     targets['scores'] = []
    #     # targets['image_id'] = torch.as_tensor(idx)
    #     targets['size'] = torch.as_tensor([h, w])
    #     targets['orig_size'] = torch.as_tensor([h, w])
    #     if pre_obj_ids is None:
    #         for id in self.all_labels[vid][idx]:
    #             for xywh in self.all_labels[vid][idx][id]:
    #                 targets['boxes'].append(xywh)
    #                 targets['obj_ids'].append(id)
    #                 targets['scores'].append(1.)
    #     else:
    #         for id in pre_obj_ids.numpy():
    #             if  id in  self.all_labels[vid][idx]:
    #                 for xywh in self.all_labels[vid][idx][id]:
    #                     targets['boxes'].append(xywh)
    #                     targets['obj_ids'].append(id)
    #                     targets['scores'].append(1.)
    #             else:
    #                 targets['boxes'].append([0 ,0 ,0 ,0])
    #                 targets['obj_ids'].append(id)
    #                 targets['scores'].append(0.0)
    #
    #     targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float32)
    #     targets['scores'] = torch.as_tensor(targets['scores'])
    #     targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
    #     # targets['boxes'][:, 2:] += targets['boxes'][:, :2]
    #     targets['img_path' ] =img_path
    #     targets['boxes_ori'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
    #
    #     return img, targets

    def read_txt(self,file_path):
        with open(file_path, 'r') as f:
            content = f.read().splitlines()
        return content





    # ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    # ratio_width, ratio_height = ratios
    #
    #
    # target = target.copy()
    # if "boxes" in target:
    #     boxes = target["boxes"]
    #     scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    #     target["boxes"] = scaled_boxes








if __name__ == '__main__':
    root=r'C:\Users\Administrator\Desktop\dance\train'

    D=PreData(root)
    seq_info=D.seq_info
    for k,info in seq_info.items():
        gt_info=D.parse_gt(info)
        for idx in range(info['seqLength']):
            pre_data,cur_data=D.process_data(info,gt_info,idx)

            frame_id=cur_data['frame_id']
            track_id=cur_data['track_id']
            boxes_xywh=cur_data['boxes_xywh_ori']
            img_path=cur_data['img_path']
            img=cur_data['img']
            resize_boxes=cur_data['boxes_xywh']

