import paddle
import numpy as np
import numpy.random as npr

from paddle.vision.transforms import Compose
from utils.utils import Rescale, Normailize, Reshape

# TODO: keep_ratio

class Collater(object):
    """"""
    def __init__(self, scales, keep_ratio=False, multiple=32):
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)
        self.keep_ratio = keep_ratio
        self.multiple = multiple
        
        self.transform = Compose([Reshape(unsqueeze=False), Normailize()])

    def __call__(self, batch):
        random_scale_inds = npr.randint(0, high=len(self.scales))
        target_size = self.scales[random_scale_inds]
        target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple) # 是 self.multiple 整数倍
        rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)

        images = [sample['image'] for sample in batch]
        bboxes = [sample['boxes'] for sample in batch]
        batch_size = len(images)
        max_width, max_height = -1, -1
        for i in range(batch_size):
            im, _ = rescale(images[i]) # 放到指定尺度
            height, width = im.shape[0], im.shape[1]
            max_width = width if width > max_width else max_width
            max_height = height if height > max_height else max_height

        # 将不同size的图放在一个模板中, 而不用整体resize
        padded_ims = np.zeros([batch_size, 3, max_height, max_width], dtype="float32")

        num_params = bboxes[0].shape[-1]   
        max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
        
        # 将不同数量的 bbox 放在一个模板中, 而不用依次传入
        padded_boxes = np.ones([batch_size, max_num_boxes, num_params], dtype="float32") * -1
        for i in range(batch_size):
            im, bbox = images[i], bboxes[i]
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = self.transform(im)
            if num_params < 9:  
                bbox[:, :4] = bbox[:, :4] * im_scale
            else:   
                bbox[:, :8] = bbox[:, :8] * np.hstack((im_scale, im_scale))
            padded_boxes[i, :bbox.shape[0], :] = bbox
        return {'image': padded_ims, 'boxes': padded_boxes}