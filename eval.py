
import argparse
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# 指标评估依赖文件
from trackeval.eval import Evaluator
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval.metrics import HOTA, CLEAR, Identity, VACE
# 模型测试依赖文件
from models.builder_model import build_model
from utils.envs.env import init_seeds
from utils.envs.torch_utils import select_device
from datasets.general_data.predata import PreData
from utils.checkpoint.file_utils import get_save_dir
from utils.checkpoint.checkpoint import load_ckpt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative




def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', type=str, default="dance", choices=['dance'],  help='data sets')
    parser.add_argument('--source', type=str, default=r'C:\Users\Administrator\Desktop\dance\train', help='file/dir/URL/glob, 0 for webcam')





    parser.add_argument('--num_classes', type=int, default=2,  help='number of boxes class')
    parser.add_argument('--model_name', type=str, default="diff_track", choices=['resnet', 'diff_track'], help='Select Model')
    parser.add_argument('--weights',  default=r'C:\Users\Administrator\Desktop\diff_track_model\diff_attention_mot-master\runs\train\exp2\diff_track_dance_20.pth', help='model path(s)')


    parser.add_argument('--img_size',  type=list, default=(540, 720), help='inference size h,w')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/eval', help='save results to project/name')

    parser.add_argument('--save_video', default=False, help='save video to project/name')
    parser.add_argument('--save_img',  default=False, help='save img to project/name')
    parser.add_argument('--show_video', default=True, help='save video to project/name')


    args = parser.parse_args()
    return args

def write_txt(text_lst,out_dir):
    '''
    每行内容为列表，将其写入text中
    '''
    file_write_obj = open(out_dir, 'w')  # 以写的方式打开文件，如果文件不存在，就会自动创建
    for text in text_lst:
        file_write_obj.writelines(text)
        file_write_obj.write('\n')
    file_write_obj.close()

    return out_dir

def model_predect(args):
    init_seeds()
    device = select_device(args.device, 1)  # 设置显卡
    save_dir = get_save_dir(args.project)

    model = build_model(args.model_name, args)

    model=load_ckpt(args.weights, model).to(device)

    model = torch.nn.DataParallel(model)  # DP 模式

    model.eval()




    data = PreData(args.source,img_size=args.img_size)
    seq_info = data.seq_info

    file_name_lst=[]
    N=0
    for k, info in seq_info.items():
        N=N+1
        if N>1:
            break
        predect_info_lst=[]

        gt_info = data.parse_gt(info)
        with tqdm(total=info['seqLength']) as pbar:
            for idx in range(info['seqLength']):
                pre_data, cur_data = data.process_data(info, gt_info, idx)
                if not pre_data: continue  # 字典为空，继续开始


                frame_id = cur_data['frame_id']
                # track_id = cur_data['track_id']
                # boxes_xywh = cur_data['boxes_xywh_ori']
                # img_path = cur_data['img_path']
                # img = cur_data['img']
                # resize_boxes = cur_data['boxes_xywh']



                pre_imgs=pre_data['img'].unsqueeze(0).to(device)
                cur_imgs = cur_data['img'].unsqueeze(0).to(device)
                pre_boxes=pre_data['boxes_xywh'].unsqueeze(0).to(device)


                # pred_logits, pred_boxes = model(pre_imgs, cur_imgs, pre_boxes)

                # outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                # print(outputs)






                # 是模型给出结果，暂时使用cur_data的结果
                trackid_lst=cur_data['track_id']
                box_lst=cur_data['boxes_xywh_ori']


                for obj_i, trackid in enumerate(trackid_lst):
                    # frame_id, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i
                    x1,y1,w,h=box_lst[obj_i]
                    x1, y1, w, h=int(x1),int(y1),int(w),int(h)

                    predect_info=str(frame_id)+','+str(trackid)+','+str(x1)+','+str(y1)+','\
                                 +str(w)+','+str(h)+',-1,-1,-1,'+str(obj_i)
                    predect_info_lst.append(predect_info)


                pbar.set_description("{}:{}".format(k, idx))
                # pbar.set_postfix(iter_all='{}||{}'.format(4, 6),  iter_epoch='{}||{}'.format(9, 10))
                pbar.update()

        file_name_lst.append(k)

        write_txt(predect_info_lst,os.path.join(save_dir,k+'.txt'))



    print('predect txt path:{}'.format(save_dir))


    return save_dir,file_name_lst



def track_eval(trackers_folder,gt_folder,output_folder,seq_info):

    config = {}

    # config['TRACKERS_FOLDER'] = ROOT + '/trackeval/data/predect_mot'
    # config['GT_FOLDER'] = ROOT + '/trackeval/data/mot17_gt'  # 给出gt路径
    # config['OUTPUT_FOLDER'] = ROOT + '/trackeval/data/out_dir'

    config['TRACKERS_FOLDER'] = trackers_folder
    config['GT_FOLDER'] = gt_folder
    config['OUTPUT_FOLDER'] = output_folder
    config['SEQ_INFO']=seq_info


    # 确定文件内gt.txt的路径，gt_folder=config['GT_FOLDER']，seq为os.listdir(gt_folder)列表
    config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt/gt.txt'
    config['CLASSES_TO_EVAL'] = ['pedestrian']  # 确定预测指标的类别

    dataset = MotChallenge2DBox(config)  # dataset_list是存放数据信息列表

    evaluator = Evaluator()

    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    metrics_list = []
    for metric in [HOTA, CLEAR, Identity, VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset, metrics_list)



















def main():

    args = parse_opt()
    predect_path,seq_info=model_predect(args)


    track_eval(predect_path,args.source, predect_path,seq_info)






if __name__ == "__main__":


    main()













