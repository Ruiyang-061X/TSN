import math
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms


class GroupRandomCrop():

    def __init__(self, size):
        self.size = size if not isinstance(size, int) else (size, size)

    def __call__(self, image_group):
        w, h = image_group[0].size
        th, tw = self.size
        if w == tw and h == th:
            image_group_out = image_group
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            image_group_out = []
            for image in image_group:
                image_group_out += [image.crop((x1, y1, x1 + tw, y1 + th))]

        return image_group_out

class GroupCenterCrop():

    def __init__(self, size):
        self.transform = transforms.CenterCrop(size)

    def __call__(self, image_group):
        image_group_out = [self.transform(image) for image in image_group]

        return image_group_out

class GroupRandomHorizontalFlip():

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, image_group):
        v = random.random()
        if v < 0.5:
            image_group_out = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in image_group]
            if self.is_flow:
                for i in range(0, len(image_group_out), 2):
                    image_group_out[i] = ImageOps.invert(image_group_out[i])
        else:
            image_group_out = image_group

        return image_group_out

class GroupNormalize():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_group):
        tensor_group_out = tensor_group
        repeat_mean = self.mean * (tensor_group.size()[0] // len(self.mean))
        repeat_std = self.std * (tensor_group.size()[0] // len(self.std))
        for t, m, s in zip(tensor_group_out, repeat_mean, repeat_std):
            t.sub_(m).div_(s)

        return tensor_group_out

class GroupScale():

    def __init__(self, size):
        self.transform = transforms.Resize(size, Image.BILINEAR)

    def __call__(self, image_group):
        image_group_out = [self.transform(image) for image in image_group]

        return image_group_out

class GroupMultiScaleCrop():

    def __init__(self, size, scale=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.size = size if not isinstance(size, int) else (size, size)
        self.scale = scale if scale is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop

    def __call__(self, image_group):
        image_size = image_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(image_size)
        image_group_crop = [image.crop((offset_w, offset_h, offset_w + crop_w, offset_h +crop_h)) for image in image_group]
        image_group_out = [image.resize(self.size, Image.BILINEAR) for image in image_group_crop]

        return image_group_out

    def _sample_crop_size(self, image_size):
        image_w, image_h = image_size
        base_size = min(image_w, image_h)
        crop_size = [int(base_size * i) for i in self.scale]
        crop_h = [self.size[1] if abs(i - self.size[1]) < 3 else i for i in crop_size]
        crop_w = [self.size[0] if abs(i - self.size[0]) < 3 else i for i in crop_size]
        pair = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pair.append((w, h))
        crop_pair = random.choice(pair)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offset = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)

        return random.choice(offset)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_h - crop_h) // 4
        h_step = (image_w - crop_w) // 4
        offset = []
        offset += [(0, 0)]
        offset += [(4 * w_step, 0)]
        offset += [(0, 4 * h_step)]
        offset += [(4 * w_step, 4 * h_step)]
        offset += [(2 * w_step, 2 * h_step)]

        if more_fix_crop:
            offset += [(0, 2 * h_step)]
            offset += [(4 * w_step, 2 * h_step)]
            offset += [(2 * w_step, 4 * h_step)]
            offset += [(2 * w_step, 0 * h_step)]

            offset += [(1 * w_step, 3 * h_step)]
            offset += [(3 * w_step, 1 * h_step)]
            offset += [(1 * w_step, 3 * h_step)]
            offset += [(3 * w_step, 3 * h_step)]

        return offset

class GroupOverSample():

    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        if scale_size is not None:
            self.transform = GroupScale(scale_size)
        else:
            self.transform = None

    def __call__(self, image_group):
        if self.transform is not None:
            image_group = self.transform(image_group)
        image_w, image_h = image_group[0].size
        crop_w, crop_h = self.crop_size
        offset = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        image_group_out = []
        for offset_w, offset_h in offset:
            normal_group = []
            flip_group = []
            for i, image in enumerate(image_group):
                image_crop = image.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
                normal_group += [image_crop]
                image_crop_flip = image_crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                if image.mode == 'L' and i % 2 == 0:
                    flip_group += [ImageOps.invert(image_crop_flip)]
                else:
                    flip_group += [image_crop_flip]
            image_group_out += normal_group
            image_group_out += flip_group

        return image_group_out

class GroupRandomSizedCrop():

    def __init__(self, size):
        self.size = size if not isinstance(size, int) else (size, size)

    def __call__(self, image_group):
        for i in range(10):
            area = image_group[0].size[0] * image_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= image_group[0].size[0] and h <= image_group[0].size[1]:
                x1 = random.randint(0, image_group[0].size[0] - w)
                y1 = random.randint(0, image_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            image_group_out = []
            for image in image_group:
                image = image.crop((x1, y1, x1 + w, y1 + h))
                image = image.resize(self.size, Image.BILINEAR)
                image_group_out += [image]
        else:
            scale = GroupScale(self.size)
            crop = GroupRandomCrop(self.size)
            image_group_out = crop(scale(image_group))

        return image_group_out

class Stack():

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, image_group):
        if image_group[0].mode == 'L':
            image_group_out = np.concatenate([np.expand_dims(image, 2) for image in image_group], 2)
        else:
            if self.roll:
                image_group_out = np.concatenate([np.array(image)[ : , : , :: -1] for image in image_group], 2)
            else:
                image_group_out = np.concatenate(image_group, 2)

        return image_group_out

class ToTorchFormatTensor():

    def __init__(self, div=True):
        self.div = div

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            image = image.float().div(255) if self.div else image.float()
        else:
            image = torch.ByteTensor(torch.BoolStorage.from_buffer(image.tobytes()))
            image = image.view(image.size[1], image.size[0], len(image.mode))
            image = image.transpose(0, 1).transpose(0, 2).contiguous()
            image = image.float().div(255) if self.div else image.float()

        return image

class IdentityTransform():

    def __call__(self, data):

        return data