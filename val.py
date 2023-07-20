

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
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--num_workers', type=int, default=0, help='load data worker number')
    
    parser.add_argument('--total_epochs', type=int, default=20, help='epoches')



    parser.add_argument('--num_classes', type=int, default=2,  help='number of boxes class')

    parser.add_argument('--model_name', type=str, default="diff_track", choices=['resnet', 'diff_track'], help='Select Model')
    parser.add_argument('--weights', nargs='+', type=str, default='False', help='model path(s)')


    parser.add_argument('--img_size',  type=list, default=[(720, 540)], help='inference size h,w')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/test', help='save results to project/name')


    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")


    args = parser.parse_args()
    return args





def main(args):
    init_seeds()
    device = select_device(args.device, args.batch_size)  # 设置显卡
    save_dir = get_save_dir(args.project, args.resume)
    # kwargs_data={'batch_size':args.batch_size}
    # train_dataset, train_loader = build_dataset(args, mode='train')
    val_dataset, val_loader = build_dataset(args, mode='val')

    model = build_model(args.model_name, args).to(device)

    model = torch.nn.DataParallel(model)  # DP 模式





    model.eval()
    with tqdm(total=len(val_loader)) as pbar:
        for iter,data in enumerate(val_loader):

            # pre_imgs, cur_images, pre_targets, cur_targets = data
            pre_imgs = data["pre_images"].to(device)            # data['pre_imgs'].to(device)  #torch.stack(data['pre_imgs'], 0).to(device)   #
            cur_images = data["cur_images"].to(device)          # data['cur_images'].to(device)
            # pre_target = pre_targets.to(device)               # data['pre_targets']
            # cur_target = cur_targets.to(device)               # data['cur_targets']
            pre_targets = convert_data2device(data["pre_targets"], args.data_type, device)
            cur_targets = convert_data2device(data["cur_targets"], args.data_type, device)



            pred_logits,pred_boxes = model(pre_imgs, cur_images, pre_targets["boxes"])

            outputs={"pred_logits":pred_logits,"pred_boxes":pred_boxes}













            # #################### 打印信息控制############
            # if (iter+1) % 100 == 0:
            #     print('\tepoch: {}|{}\tloss:{}'.format(epoch + 1, iter + 1, np_loss))
            # pbar.set_description("epoch {}|{}".format(args.total_epochs, epoch + 1))
            # pbar.set_postfix(iter_all='{}||{}'.format(max_iter, cur_iter),
            #                  iter_epoch='{}||{}'.format(len(train_loader), iter + 1), loss=np_loss)
            # pbar.update()









if __name__ == "__main__":

    args = parse_opt()
    main(args)







