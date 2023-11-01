
import cv2
import argparse
import os
import sys
import torch
from pathlib import Path
import numpy as np 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from datasets.builder_dataset import build_dataset
# import datasets
from models.builder_model import build_model
from utils.envs.env import init_seeds
from utils.envs.torch_utils import select_device
from tqdm import tqdm
from utils.checkpoint.file_utils import get_save_dir
from utils.checkpoint.checkpoint import load_ckpt
from datasets.general_data.predata import PreData


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default="dance", choices=['dance'],  help='data sets')
    parser.add_argument('--source', type=str, default="/Data/cj/test_redis/DanceTrack/val", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--num_classes', type=int, default=2,  help='number of boxes class')
    parser.add_argument('--model_name', type=str, default="diff_track", choices=['resnet', 'diff_track'], help='Select Model')
    parser.add_argument('--weights',  default='/Data/cj/test_redis/DanceTrack/diff_attention_mot/runs/train/exp5/best_79.pth', help='model path(s)')


    parser.add_argument('--img_size',  type=list, default=(720, 540), help='inference size h,w')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/test', help='save results to project/name')

    parser.add_argument('--save_video', default=True, help='save video to project/name')
    parser.add_argument('--save_img',  default=True, help='save img to project/name')
    parser.add_argument('--show_video', default=False, help='save video to project/name')


    args = parser.parse_args()
    return args


def main(args):
    init_seeds()
    device = select_device(args.device, 1)  # 设置显卡
    save_dir = get_save_dir(args.project)

    model = build_model(args.model_name, args)

    model=load_ckpt(args.weights, model).to(device)

    model = torch.nn.DataParallel(model)  # DP 模式

    model.eval()

    data = PreData(args.source,img_size=args.img_size)
    seq_info = data.seq_info

    for k, info in seq_info.items():

        if args.save_video:
            video_path=os.path.join(save_dir,k+'.mp4')
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MPEG'), 25, (1920, 1080))
        gt_info = data.parse_gt(info)
        with tqdm(total=info['seqLength']) as pbar:
            for idx in range(info['seqLength']):
                pre_data, cur_data = data.process_data(info, gt_info, idx)
                if not pre_data: continue  # 字典为空，继续开始
                # frame_id = cur_data['frame_id']
                # track_id = cur_data['track_id']
                # boxes_xywh = cur_data['boxes_xywh_ori']
                # img_path = cur_data['img_path']
                # img = cur_data['img']
                # resize_boxes = cur_data['boxes_xywh']
                pre_imgs=pre_data['img'].unsqueeze(0).to(device)
                cur_imgs = cur_data['img'].unsqueeze(0).to(device)
                pre_boxes=pre_data['boxes_xywh'].unsqueeze(0).to(device)
                pred_logits, pred_boxes = model(pre_imgs, cur_imgs, pre_boxes)

                outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                # print(outputs)

                pbar.set_description("{}:{}".format(k, idx))
                # pbar.set_postfix(iter_all='{}||{}'.format(4, 6),  iter_epoch='{}||{}'.format(9, 10))
                pbar.update()

                # 是模型给出结果，暂时使用cur_data的结果
                # trackid_lst=cur_data['track_id']
                # box_lst=cur_data['boxes_xywh_ori']
                # box_lst[:,2]=box_lst[:,0]+box_lst[:,2]
                # box_lst[:, 3] = box_lst[:, 1] + box_lst[:, 3]
                
                box_lst = pred_boxes.cpu().squeeze(0).detach().numpy()
                # box_lst = pred_boxes.cpu().squeeze(0).detach().numpy()
                trackid_lst = np.arange(len(box_lst))
                

                if args.save_video or args.save_img or args.show_video:
                    from utils.vision.visual import draw_img,show_video
                    from utils.checkpoint.file_utils import build_dir
                    img=cv2.imread(cur_data['img_path'])
                    h,w,_ = img.shape
                    box_lst[:,::2] = box_lst[:,0::2]*w
                    box_lst[:,1::2] = box_lst[:,1::2]*h
                    img=draw_img(img, trackid_lst, box_lst, score_lst=None,
                             bbox_color='green',
                             text_color='green',
                             thickness=1,
                             font_scale=0.5
                             )
                    if args.show_video:
                        show_video(img, width=720, height=540)
                    if args.save_img:
                        save_img_path=os.path.join(save_dir,k)
                        build_dir(save_img_path)
                        img_name=cur_data['img_name']
                        cv2.imwrite(os.path.join(save_img_path,img_name),img)

                    if args.save_video:
                        img = cv2.resize(img, (1920, 1080))  # 将图片转换为1280*720
                        video.write(img)  # 写入视频


        if args.save_video:
            video.release()


























if __name__ == "__main__":

    args = parse_opt()
    main(args)







