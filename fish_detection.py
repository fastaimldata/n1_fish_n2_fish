import numpy as np
import pandas as pd
import json
import os
import pickle
from typing import List, Dict
from collections import namedtuple
from skimage.transform import SimilarityTransform
from sklearn.model_selection import train_test_split
import dataset
from dataset import SPECIES, CLASSES

INPUT_ROWS = 720
INPUT_COLS = 360
input_shape = (INPUT_ROWS, INPUT_COLS, 3)
NUM_CLASSES = len(CLASSES)

FishDetection = namedtuple('FishDetection', ['clip_name', 'frame', 'fish_number', 'x1', 'y1', 'x2', 'y2', 'class_id'])
RulerPoints = namedtuple('RulerPoints', ['x1', 'y1', 'x2', 'y2'])


class FishDetectionDataset:
    def __init__(self):
        self.video_clips = dataset.video_clips()

        cache_fn = '../output/fish_detection.pkl'
        try:
            # raise FileNotFoundError
            self.detections = pickle.load(open(cache_fn, 'rb'))  # type: Dict[FishDetection]
        except FileNotFoundError:
            self.detections = self.load()  # type: Dict[FishDetection]
            pickle.dump(self.detections, open(cache_fn, 'wb'))

        self.train_clips, self.test_clips = train_test_split(sorted(self.detections.keys()),
                                                             test_size=0.1,
                                                             random_state=42)

        self.nb_train_samples = sum([len(self.detections[clip]) for clip in self.train_clips])
        self.nb_test_samples = sum([len(self.detections[clip]) for clip in self.test_clips])

        self.ruler_points = {}
        ruler_points = pd.read_csv('../output/ruler_points.csv')
        for _, row in ruler_points.iterrows():
            self.ruler_points[row.clip_name] = RulerPoints(x1=row.ruler_x0, y1=row.ruler_y0, x2=row.ruler_x1, y2=row.ruler_y1)

    def load(self):
        detections = {}
        ds = pd.read_csv('../input/N1_fish_N2_fish_-_Training_set_annotations.csv')

        species = ds.as_matrix(columns=['species_'+s for s in SPECIES])
        cls_column = np.argmax(species, axis=1)+1
        cls_column[np.max(species, axis=1) == 0] = 0

        for row_id, row in ds.iterrows():
            clip_name = row.video_id
            if clip_name not in detections:
                detections[clip_name] = []

            detections[clip_name].append(
                FishDetection(
                    clip_name=clip_name,
                    frame=row.frame,
                    fish_number=row.fish_number,
                    x1=row.x1, y1=row.y1,
                    x2=row.x2, y2=row.y2,
                    class_id=int(cls_column[row_id])
                )
            )
        return detections

    def transform_for_clip(self, clip_name, dst_w=720, dst_h=360, points_random_shift=0):
        points = self.ruler_points[clip_name]

        ruler_points = np.array([[points.x1, points.y1], [points.x2, points.y2]])
        img_points = np.array([[dst_w * 0.1, dst_h / 2], [dst_w * 0.9, dst_h / 2]])

        if points_random_shift > 0:
            img_points += np.random.uniform(-points_random_shift, points_random_shift, (2, 2))

        tform = SimilarityTransform()
        tform.estimate(dst=ruler_points, src=img_points)

        return tform
