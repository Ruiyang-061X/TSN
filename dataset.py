import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VideoRecord():

    def __init__(self, path, n_frame, label):
        self.path = path
        self.n_frame = n_frame
        self.label = label

class TSNDataset(Dataset):

    def __init__(self, video_list_path, modality='RGB', train=True, n_segment=3, new_length=1, random_shift=True, tranform=None):
        self.video_list_path = video_list_path
        self.modality = modality
        self.train = train
        self.n_segment = n_segment
        self.new_length = new_length
        self.random_shift = random_shift
        self.tranform = tranform
        if self.modality == 'RGBDiff':
            self.new_length += 1
        self._parse_video_list()

    def _parse_video_list(self):
        self.video_record_list = [VideoRecord(i.strip().split(' ')[0], int(i.strip().split(' ')[1]), int(i.strip().split(' ')[2])) for i in open(self.video_list_path)]

    def _get_segment_indice(self, video_record):
        if video_record.n_frame > self.n_segment + self.new_length - 1:
            stride = (video_record.n_frame - self.new_length + 1) / float(self.n_segment)
            indice = np.array([int(stride * i + stride / 2.0) for i in range(self.n_segment)])
        else:
            indice = np.zeros((self.n_segment), dtype=np.int)

        return indice + 1

    def _get_random_shift_segment_indice(self, video_record):
        duration = (video_record.n_frame - self.new_length + 1) // self.n_segment
        if duration > 0:
            indice = np.multiply(range(self.n_segment), duration) + np.random.randint(duration, size=self.n_segment)
        elif video_record.n_frame > self.n_segment:
            indice = np.sort(np.random.randint(video_record - self.new_length + 1, size=self.n_segment))
        else:
            indice = np.zeros((self.n_segment))
        indice = indice.astype(np.int)

        return indice + 1

    def _get_validation_segment_indice(self, video_record):
        stride = (video_record.n_frame - self.new_length + 1) / float(self.n_segment)
        indice = np.array([int(stride * i + stride / 2.0) for i in range(self.n_segment)])

        return indice + 1

    def _load_image(self, path, indice):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':

            return [Image.open(os.path.join(path, 'img_{:05d}.jpg'.format(indice))).convert('RGB')]
        elif self.modality == 'Flow':
            flow_x = Image.open(os.path.join(path, 'flow_x_{:05d}.jpg'.format(indice))).convert('L')
            flow_y = Image.open(os.path.join(path, 'flow_y_{:05d}.jpg'.format(indice))).convert('L')

            return [flow_x, flow_y]

    def __getitem__(self, index):
        video_record = self.video_record_list[index]
        if self.train:
            segment_indice = self._get_random_shift_segment_indice(video_record) if self.random_shift else self._get_segment_indice(video_record)
        else:
            segment_indice = self._get_validation_segment_indice(video_record)

        image = []
        for i in segment_indice:
            for j in range(self.new_length):
                tmp = self._load_image(video_record.path, i)
                image += tmp
                if i < video_record.n_frame:
                    i += 1

        if self.tranform is not None:
            image = self.tranform(image)

        return image, video_record.label

    def __len__(self):

        return len(self.video_record_list)