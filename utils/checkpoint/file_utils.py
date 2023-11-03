

import os



def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_save_dir(out_dir,resume=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    max_i=0
    for name in os.listdir(out_dir):
       N=len(name)
       if 'exp' in name and N>3:
           v=int(name[3:])
           if v>max_i:  max_i=v
    name='exp'+str(max_i+1)  if not resume else 'exp'+str(max_i)
    save_dir=os.path.join(out_dir,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir



def build_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    return out_dir





