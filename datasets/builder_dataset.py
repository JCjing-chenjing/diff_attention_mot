from torch.utils.data import DataLoader
from datasets.dance.dance import DanceDataset, collate_fn


# 构建数据方法
def build_dataset(args, mode='train'):
    # opts.data_type, opts.source
    #data_name=None,data_dir=None
    if args.data_type=='dance':
        batch_size = args.batch_size
        num_workers = args.num_workers
        data_set= DanceDataset(args, image_set=mode)
        # data_loader = DataLoader(dataset=data_set, batch_size=batch_size, #collate_fn=mot_collate_fn,
        #                          shuffle=True, num_workers=num_workers,  drop_last=False)
        data_loader = DataLoader(dataset=data_set, batch_size=batch_size, collate_fn = collate_fn,
                                 shuffle=True, num_workers=num_workers,  drop_last=False)

    else:
        raise ValueError ("failed of loading data,check data format or data_dir ...")

    return data_set,data_loader


def convert_data2device(target,data_name,device):


    if data_name=='dance':
        for k in list(target.keys()):

            target[k]=target[k].to(device)
    else:
        target=target.to(device)

    return target







if __name__ == "__main__":

    data_dir = r"C:\Users\Administrator\Desktop\dance"
    dataset_train,dataloader_train=build_dataset('dance',data_dir,mode='train')
    dataset_val, dataloader_val = build_dataset('dance', data_dir, mode='val')
    for data in dataloader_train:
        pre_imgs = data['pre_imgs']
        cur_images = data['cur_images']
        pre_target = data['pre_targets']
        cur_target = data['cur_targets']
        print(pre_imgs[0].shape, cur_images[1].shape, pre_target, cur_target)











