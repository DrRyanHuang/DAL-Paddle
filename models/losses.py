import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps as rbbx_overlaps
# from utils.overlaps_cuda.rbbox_overlaps import rbbx_overlaps

def fuck_paddle_idx(inp, mask):
    
    if len(inp.shape)==2:
        m, n = inp.shape
        idx = paddle.to_tensor(np.arange(m, dtype="int32")[mask.numpy()])
        
        if idx.shape[0] == 0:
            return paddle.to_tensor(np.zeros(shape=(0, n)))
        
        return paddle.index_select(inp, idx, axis=0)
    
    elif len(inp.shape)==1:
        m = len(inp)
        idx = paddle.to_tensor(np.arange(m, dtype="int32")[mask.numpy()])
        
        if idx.shape[0] == 0:
            return paddle.to_tensor(np.zeros(shape=(0,)))
        
        return paddle.index_select(inp, idx, axis=0)
    
    
def boolidx2tensor(mask):
    assert len(mask.shape)==1
    m = len(mask)
    idx = paddle.to_tensor(np.arange(m, dtype="int32")[mask.numpy()])
    return idx


def xyxy2xywh_a(query_boxes):
    out_boxes = query_boxes.copy()
    out_boxes[:, 0] = (query_boxes[:, 0] + query_boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (query_boxes[:, 1] + query_boxes[:, 3]) * 0.5
    out_boxes[:, 2] = query_boxes[:, 2] - query_boxes[:, 0]
    out_boxes[:, 3] = query_boxes[:, 3] - query_boxes[:, 1]
    return out_boxes


# cuda_overlaps
class IntegratedLoss(nn.Layer):
    def __init__(self, alpha=0.25, gamma=2.0, func='smooth'):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss

    def forward(self, classifications, regressions, anchors, refined_achors, annotations, \
                md_thres=0.5, mining_param=(1, 0., -1), ref=False):
        
        das = True
        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        alpha, beta, var = mining_param
        # import ipdb;ipdb.set_trace()
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            # Paddle 不支持直接这样索引
            # bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            
            # Paddle2.3 依旧不支持, 焯
            # bbox_annotation = paddle.masked_select(bbox_annotation, 
            #                                        bbox_annotation[:, -1] != -1)
            
            # # Paddle2.3 mask 堆叠2次不支持 data_type[bool] ......
            # # 我只能用这种比较蠢的方式....
            # mask = bbox_annotation[:, -1] != -1
            # _, n = bbox_annotation.shape
            # mask_int = paddle.stack([mask.cast("int")]*n, 
            #                         axis=1)
            # mask = mask_int.cast("bool")
            # bbox_annotation = paddle.masked_select(bbox_annotation, 
            #                                        mask)
            # bbox_annotation = bbox_annotation.reshape((-1, n))
            # # 这样做完 bbox_annotation 错误
            # # https://github.com/PaddlePaddle/Paddle/issues/45007
            
            

            
            
            if bbox_annotation.shape[0] == 0:
            # if idx.shape[0] == 0:
                
                # cls_losses.append(torch.tensor(0).float().cuda())
                # reg_losses.append(torch.tensor(0).float().cuda())
                
                cls_losses.append(paddle.to_tensor(0.0))
                reg_losses.append(paddle.to_tensor(0.0))
                continue
            
            mask = bbox_annotation[:, -1] != -1
            bbox_annotation = fuck_paddle_idx(bbox_annotation, mask)
            # m, n = bbox_annotation.shape
            # idx = paddle.masked_select(paddle.arange(m, dtype="int32"), 
            #                             mask)
            # idx = paddle.to_tensor(np.arange(m, dtype="int32")[mask.numpy()])
            
            # bbox_annotation = paddle.index_select(bbox_annotation, idx, axis=0)
            
            
            
            classification = paddle.clip(classification, 1e-4, 1.0 - 1e-4)
            sa = rbbx_overlaps(
                xyxy2xywh_a(anchors[j, :, :].cpu().numpy()),
                xyxy2xywh_a(bbox_annotation[:, :-1].cpu().numpy()),
            )
            if not paddle.is_tensor(sa):
                # import ipdb;ipdb.set_trace()
                sa = paddle.to_tensor(sa)
            if var != -1:
                fa = rbbx_overlaps(
                    xyxy2xywh_a(refined_achors[j, :, :].cpu().numpy()),
                    xyxy2xywh_a(bbox_annotation[:, :-1].cpu().numpy()),
                )
                if not paddle.is_tensor(fa):
                    fa = paddle.to_tensor(fa)

                if var == 0:
                    md = paddle.abs((alpha * sa + beta * fa))
                else:
                    md = paddle.abs((alpha * sa + beta * fa) - paddle.abs(fa - sa)**var)
            else:
                das = False
                md = sa
            
            iou_max, iou_argmax = paddle.max(md, axis=1), paddle.argmax(md, axis=1)
           
            # positive_indices = paddle.greater_equal(iou_max, md_thres)
            positive_indices = iou_max >= md_thres

             
            max_gt, argmax_gt = md.max(0), md.argmax(0)
            # import ipdb;ipdb.set_trace(context = 15)
            if (max_gt < md_thres).any():
                
                # Paddle无法直接索引，这个操作好窒息
                # positive_indices[argmax_gt[max_gt < md_thres]]=1
                
                argmax_gt_mask = max_gt < md_thres
                positive_indices_idx = fuck_paddle_idx(argmax_gt, argmax_gt_mask)
                positive_indices[ positive_indices_idx.numpy().tolist() ] = 1
                
              
            # matching-weight
            if das:
                # pos = md[positive_indices]
                pos = fuck_paddle_idx(md, positive_indices)
                # pos_mask = paddle.greater_equal(pos, md_thres)
                pos_mask = pos >= md_thres
                max_pos, armmax_pos = pos.max(0), pos.argmax(0)
                nt = md.shape[1]
                for gt_idx in range(nt):
                    pos_mask[armmax_pos[gt_idx], gt_idx] = 1
                    
                left = (1 - max_pos).tile((len(pos),1))
                comp = paddle.where(pos_mask, left, pos)
                matching_weight = comp + pos
            # import ipdb; ipdb.set_trace(context = 15)

            # cls loss
            cls_targets = paddle.ones(classification.shape) * -1
            # left_idx = paddle.less_than(iou_max, md_thres - 0.1)
            left_idx = iou_max < md_thres - 0.1
            cls_targets[left_idx, :] = 0
            num_positive_anchors = positive_indices.sum()
            # assigned_annotations = bbox_annotation[iou_argmax, :]
            assigned_annotations = bbox_annotation[iou_argmax]
            
            cls_targets[positive_indices, :] = 0
            # right = assigned_annotations[positive_indices, -1].long()
            right = assigned_annotations[:, -1][positive_indices].cast("int")
            
            cls_targets[positive_indices, right] = 1
            alpha_factor = paddle.ones(cls_targets.shape) * self.alpha
            alpha_factor = paddle.where(paddle.equal(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = paddle.where(paddle.equal(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * paddle.pow(focal_weight, 
                                                     self.gamma)
            bin_cross_entropy = -(cls_targets * paddle.log(classification+1e-6) + \
                                 (1.0 - cls_targets) * paddle.log(1.0 - classification+1e-6))
            if das :
                soft_weight = paddle.zeros(classification.shape)
                soft_weight = paddle.where(paddle.equal(cls_targets, 0.), 
                                           paddle.ones_like(cls_targets), 
                                           soft_weight)
                
                # right = assigned_annotations[positive_indices, -1].long()
                right = fuck_paddle_idx(assigned_annotations[:, -1], positive_indices).cast("int")
                # soft_weight[positive_indices, right] = (matching_weight.max(1) + 1)
                soft_weight[boolidx2tensor(positive_indices), right] = (matching_weight.max(1) + 1)
                
                cls_loss = focal_weight * bin_cross_entropy * soft_weight
            else:
                cls_loss = focal_weight * bin_cross_entropy 
            cls_loss = paddle.where(
                cls_targets != -1.0,
                # paddle.not_equal(cls_targets, -1.0), 
                cls_loss, 
                paddle.zeros(cls_loss.shape))
            cls_losses.append(cls_loss.sum() / paddle.clip(num_positive_anchors.cast("float32"), min=1.0))
            # reg loss
            if positive_indices.sum() > 0:
                # all_rois = anchors[j, positive_indices, :]
                # 其实就是：all_rois = anchors[j][positive_indices]
                all_rois = fuck_paddle_idx(anchors[j], positive_indices)
                # gt_boxes = assigned_annotations[positive_indices, :]
                # 其实就是：gt_boxes = assigned_annotations[positive_indices]
                gt_boxes = fuck_paddle_idx(assigned_annotations, positive_indices)
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)
                
                # first_param = regression[positive_indices, :]
                first_param = fuck_paddle_idx(regression, positive_indices)
                if das:
                    reg_loss = self.criteron(first_param, 
                                             reg_targets, 
                                             weight = matching_weight)
                else:
                    reg_loss = self.criteron(first_param, reg_targets)
                reg_losses.append(reg_loss)

                if not paddle.isfinite(reg_loss) :
                    import ipdb; ipdb.set_trace()
                k=1
            else:
                reg_losses.append(paddle.to_tensor(0.0))
        loss_cls = paddle.stack(cls_losses).mean(axis=0, keepdim=True)
        loss_reg = paddle.stack(reg_losses).mean(axis=0, keepdim=True)
        return loss_cls, loss_reg

    
def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight = None):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = paddle.abs(inputs - targets)
    if  weight is  None:
        loss = paddle.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = paddle.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1).unsqueeze(1).tile((1,5))
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = paddle.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = paddle.where(
        diff < beta, 
        alpha / b * (b * diff + 1) * paddle.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()

