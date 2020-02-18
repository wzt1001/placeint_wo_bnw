from __future__ import print_function, division

import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon

# class RandomCrop(object):
#     """Randomly Crop the frames in a clip."""

#     def __init__(self, output_size):
#         """
#             Args:
#               output_size (tuple or int): Desired output size. If int, square crop
#               is made.
#         """
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size

#     def __call__(self, clip):
#         h, w = clip.size()[2:]
#         new_h, new_w = self.output_size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         clip = clip[:, :, top : top + new_h, left : left + new_w]
#         return clip


class GeneralVideoDataset():
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        video_file,
        channels,
        batch_size,
        fps,
        frames_per_detect,
        effective_zone,
        w=1280,
        h=720
    ):
        """
        Args:
            video_file (string): Path to the clipsList file with labels.
            root_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            channels: Number of channels of frames
            time_depth: Number of frames to be loaded in a sample
            x_size, y_size: Dimensions of the frames
            mean: Mean value of the training set videos over each channel
        """
        self.video_file = video_file
        self.channels = channels
        self.batch_size = batch_size
        self.cap = cv2.VideoCapture(self.video_file)
        self.fps = fps
        self.frames_per_detect = frames_per_detect
        self.k = 0
        # x, y for effective zones
        self.w = w
        self.h = h
        self.effective_x, self.effective_y = [int(i["x"] * self.w / 640) for i in effective_zone], [int(i["y"] * self.h / 480) for i in effective_zone]

        self.mask = self.get_mask()

        self.min_x = np.min(np.where(self.mask==True)[1])
        self.min_y = np.min(np.where(self.mask==True)[0])
        self.max_x = np.max(np.where(self.mask==True)[1])
        self.max_y = np.max(np.where(self.mask==True)[0])

    def __len__(self):
        return len(self.clipsList)

    def get_new_batch(self):
        # Open the video file
        # frames = torch.FloatTensor(
        #     self.channels, self.time_depth, self.x_size, self.y_size
        # )
        seq = []

        # frame = torch.from_numpy(frame)
        # frame = frame.permute(2, 0, 1)
        # frames[:, f, :, :] = frame
        # .transpose(2,0,1)

        batch_cnt = 0

        while (batch_cnt < self.batch_size):
            try:
                valid, frame = self.cap.read()
            except:
                continue
            if valid == False:
                break

            if self.k % self.fps % self.frames_per_detect == 0:
                masked_frame = frame.copy()

                # hide unnecessary zones using black pixels
                masked_frame[np.where(self.mask==0)] = [0, 0, 0]

                # crop the image
                masked_frame = masked_frame[self.min_y:self.max_y, self.min_x:self.max_x, :]

                seq.append(masked_frame.transpose(2,0,1))
                batch_cnt += 1
            self.k += 1

        # data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        return torch.from_numpy(np.asarray(seq) / 255.0), valid

    def get_mask(self):

        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        rr, cc = polygon(self.effective_y, self.effective_x)
        mask[rr, cc] = 1
        black = np.zeros((self.h, self.w, 3))

        # ======  below adds new pixels for upper boundary ======

        mask_shift = mask.copy()
        mask_shift[1:self.h, :] = mask_shift[0:self.h-1, :]
        f = mask-mask_shift == 1

        for col in range(mask.shape[1]):
            col_val = mask[:,col]
            upper_boundary = np.where(f[:,col] == True)[0]
            if len(upper_boundary) == 0:
                continue
            for bound in upper_boundary:
                # adding pixel subject to changes
                add_pixel = int(bound/1.6)
                mask[max(bound - 1 - add_pixel, 0):bound, col] = 1

        return mask




