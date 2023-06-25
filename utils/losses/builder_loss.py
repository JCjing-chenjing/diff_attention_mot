# _*_ coding: utf-8 _*_
"""
Time:     2023/5/17 17:02
Author:   jun tang(owen)
File:     builder_loss.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch


def build_loss(loss_name,args=None, **kwargs):


    criteria = None
    if loss_name == 'detr_loss':
        from utils.losses.detr_loss.matcher import HungarianMatcher
        from utils.losses.detr_loss.loss import SetCriterion


        matcher = HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                   cost_giou=args.set_cost_giou)
        losses = ['labels', 'boxes', 'cardinality']

        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        if args.masks:
            weight_dict["loss_mask"] = 1
            weight_dict["loss_dice"] = 1
        # TODO this is a hack

        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        criteria = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)

    elif loss_name=='general_loss':
        from utils.losses.general_loss import loss








    return criteria
