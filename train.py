

import argparse
import os
import sys
import torch
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from datasets.builder_dataset import build_dataset
from models.builder_model import build_model
from utils.optim.builder_optimizer import build_optimizer
from utils.envs.env import init_seeds
from utils.envs.torch_utils import select_device

from tqdm import tqdm
from utils.losses.builder_loss import build_loss
from datasets.builder_dataset import convert_data2device
from utils.checkpoint.file_utils import get_save_dir
from utils.checkpoint.checkpoint import save_ckpt


def parse_opt():
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_type', type=str, default="dance", choices=['dance'],  help='data sets')
    parser.add_argument('--source', type=str, default=r"C:\Users\Administrator\Desktop\dance", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'], help='data ')
    parser.add_argument('--batch_size', type=int, default=2, help='total batch size for all GPUs')   #32
    parser.add_argument('--num_workers', type=int, default=0, help='load data worker number')        #16
    
    parser.add_argument('--total_epochs', type=int, default=200, help='epoches')


    parser.add_argument('--optim_name', type=str, default="sgd", choices=['sgd'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='lr')
    parser.add_argument('--num_classes', type=int, default=2,  help='number of boxes class')

    parser.add_argument('--model_name', type=str, default="diff_track", choices=['resnet', 'diff_track'], help='Select Model')
    parser.add_argument('--weights', nargs='+', type=str, default='False', help='model path(s)')
    parser.add_argument('--resume', default=False, help='model path(s)')

    parser.add_argument('--img_size',  type=list, default=[(720, 540)], help='inference size h,w')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  #'1,2'
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save results to project/name')

    parser.add_argument('--loss_name', type=str, default="general_loss", choices=['general_loss'], help='Select loss')
    parser.add_argument('--save_period', type=int, default=1, help='eeight storage step size')

    parser.add_argument('--dec_layers', default=6, type=int,  help="Number of decoding layers in the transformer")


    # 目标检测参数
    parser.add_argument('--is_det',  default=True,  help='若为True表示打开目标检测')

    args = parser.parse_args()
    return args



def targert_track2obj(targets):
    '''
    targets中包含labels与boxes字典，且为张量
    target1 = {'boxes':torch.rand((5,4)),'labels':torch.tensor([1,3,2,1,2])}
    target2 = {'boxes': torch.rand((3, 4)), 'labels': torch.tensor([1, 1, 2])}
    target = [target1, target2]
    '''
    targets_detect=[]
    for i ,label in enumerate(targets['labels']):
        index = label.bool()  # 大于0才为True，自然排除为0的背景

        boxes_det=targets['boxes'][i][index]
        labels_det = targets['labels'][i][index]
        # labels_det=torch.(labels_det)
        targets_detect.append({'boxes':boxes_det,'labels':labels_det.long()})

    return targets_detect

def main(args):
    init_seeds()
    device = select_device(args.device, args.batch_size)  # 设置显卡
    save_dir = get_save_dir(args.project, args.resume)
    # kwargs_data={'batch_size':args.batch_size}
    train_dataset, train_loader = build_dataset(args, mode='train')
    # dataset_val, val_loader = build_dataset(opts.data_type, opts.source, mode='val',**kwargs_data)
    model = build_model(args.model_name, args).to(device)

    # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3], output_device=0)
    model = torch.nn.DataParallel(model)  # DP 模式
    optimizer = build_optimizer(args.optim_name, model, args.lr)


    criterion=build_loss(args.loss_name,args)


    # 构建目标检测的loss
    if args.is_det:
        from utils.losses.detr_loss.matcher import HungarianMatcher
        from utils.losses.detr_loss.loss import SetCriterion
        num_classes = args.num_classes   #  类别+1
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)  # 二分匹配不同任务分配的权重
        losses = ['labels', 'boxes', 'cardinality']  # 计算loss的任务
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}  # 为dert最后一个设置权重
        criterion_det = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)










    max_iter=args.total_epochs*len(train_loader)
    best_score=9999
   
    for epoch in range(args.total_epochs):
        batch_loss = 0
        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            for iter, data in enumerate(train_loader):
                # pre_imgs, cur_images, pre_targets, cur_targets = data
                pre_imgs = data["pre_images"].to(device)            #data['pre_imgs'].to(device)  #torch.stack(data['pre_imgs'], 0).to(device)   #
                cur_images = data["cur_images"].to(device)          #data['cur_images'].to(device)

                pre_targets = convert_data2device(data["pre_targets"], args.data_type, device)
                cur_targets = convert_data2device(data["cur_targets"], args.data_type, device)





                outputs = model(pre_imgs, cur_images, pre_targets["boxes"])


                outputs_track={"pred_logits":outputs['pred_logits'],"pred_boxes":outputs['pred_boxes']}

                loss_track = criterion(outputs_track, cur_targets)

                loss_det = 0.0
                if args.is_det:
                    outputs_detect = outputs['pred_detect']
                    targets = targert_track2obj(cur_targets)
                    loss_det = criterion_det(outputs_detect, targets)

                loss = loss_track+loss_det/10

                optimizer.zero_grad()


                loss.backward()
                optimizer.step()
                np_loss = loss.detach().cpu().numpy()
                cur_iter = (epoch * len(train_loader)) + iter + 1

                # scheduler.step()
                batch_loss += np_loss
                # #################### 打印信息控制############
                if (iter+1) % 100 == 0:
                    print('\tepoch: {}|{}\tloss:{}'.format(epoch + 1, iter + 1, batch_loss/cur_iter))
                pbar.set_description("epoch {}|{}".format(args.total_epochs, epoch + 1))
                pbar.set_postfix(iter_all='{}||{}'.format(max_iter, cur_iter),
                                 iter_epoch='{}||{}'.format(len(train_loader), iter + 1), loss=np_loss)
                pbar.update()



            save_ckpt(os.path.join(save_dir, 'last.pth'), model, optimizer,  epoch, best_score)
            if best_score > batch_loss:
                best_score = batch_loss
                # pth_name = str(args.model_name) + "_" + str(args.data_type) + '_' + "be" + '.pth'
                save_ckpt(os.path.join(save_dir, f'best_{epoch}.pth'), model, optimizer, epoch, best_score)




            

if __name__ == "__main__":

    args = parse_opt()
    main(args)







