from collections import namedtuple

import numpy as np
import skimage
import skimage.transform
from typing import Union
import sys
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from scipy.misc import imread, imresize
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Input, Activation, BatchNormalization, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from PIL import Image
from copy import copy

from multiprocessing.pool import ThreadPool
from typing import List, Dict

import pickle, os, random
import utils
import scipy.misc

import img_augmentation

import dataset
from dataset import SPECIES, CLASSES
import fish_detection

import densenet161
import densenet121

EXTRA_LABELS_BASE_DIR = '../output/ruler_crops_batch_labeled'
EXTRA_LABELS_BATCHES = ['0', '100', '400', '500']

CROP_WIDTH = 720
CROP_HEIGHT = 360

INPUT_ROWS = 300
INPUT_COLS = 300
INPUT_SHAPE = (INPUT_ROWS, INPUT_COLS, 3)
SPECIES_CLASSES = CLASSES

COVER_CLASSES = ['no fish', 'hand over fish', 'fish clear']
CLASS_NO_FISH_ID = 0
CLASS_HAND_OVER_ID = 1
CLASS_FISH_CLEAR_ID = 2


def build_model_densenet_161():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = densenet161.DenseNet(
        img_input=img_input,
        reduction=0.5,
        weights_path='../input/densenet161_weights_tf.h5',
        classes=1000)
    base_model.layers.pop()
    base_model.layers.pop()

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)
    # output = concatenate([species_dense, cover_dense], axis=0)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_densenet_121():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = densenet121.DenseNet(
        img_input=img_input,
        reduction=0.5,
        weights_path='../input/densenet121_weights_tf.h5',
        classes=1000)
    base_model.layers.pop()
    base_model.layers.pop()

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)
    # output = concatenate([species_dense, cover_dense], axis=0)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


FishClassification = namedtuple('FishClassification', ['video_id',
                                                       'frame',
                                                       'x', 'y', 'w',
                                                       'species_class',
                                                       'cover_class'])

SSDDetection = namedtuple('SSDDetection', ['video_id',
                                           'frame',
                                           'x', 'y', 'w', 'h',
                                           'class_id', 'confidence'
                                           ])

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])


class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 fish_classification: FishClassification,
                 saturation=0.5, contrast=0.5, brightness=0.5,  # 0.5  - no changes, range 0..1
                 scale_rect_x=1.0, scale_rect_y=1.0,
                 shift_x_ratio=0.0, shift_y_ratio=0.0,
                 angle=0.0,
                 hflip=False,
                 vflip=False):
        self.angle = angle
        self.shift_x_ratio = shift_x_ratio
        self.shift_y_ratio = shift_y_ratio
        self.scale_rect_y = scale_rect_y
        self.scale_rect_x = scale_rect_x
        self.fish_classification = fish_classification
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.cache_img = False

        w = np.clip(fish_classification.w + 64, 200, 360)
        x = fish_classification.x
        y = np.clip(fish_classification.y, CROP_HEIGHT/2 - 64, CROP_HEIGHT/2 + 64)

        if fish_classification.cover_class == CLASS_NO_FISH_ID:
            w = random.randrange(200, 360)
            x = random.randrange(200, CROP_WIDTH-200)
            y = random.randrange(CROP_HEIGHT/2 - 64, CROP_HEIGHT/2 + 64)

        self.rect = Rect(x=x-w/2, y=y-w/2, w=w, h=w)

    def __lt__(self, other):
        return True

    def __str__(self):
        return dataset.CLASSES[self.fish_classification.species_class] + ' ' + str(self.__dict__)


def load_ssd_detection(video_id, frame_id) -> SSDDetection:
    fn = '../output/predictions_ssd_roi2/vgg_41/{}/{:04}.npy'.format(video_id, frame_id + 1)
    try:
        results = np.load(fn)
    except FileNotFoundError:
        print("ssd prediction not found:", fn)
        return None

    if len(results) == 0:
        return None

    det_label = results[:, 0]
    det_conf = results[:, 1]
    det_xmin = results[:, 2]
    det_ymin = results[:, 3]
    det_xmax = results[:, 4]
    det_ymax = results[:, 5]
    top_indices_conf = sorted([(conf, i) for i, conf in enumerate(det_conf) if conf >= 0.1], reverse=True)
    if len(top_indices_conf) == 0:
        return None

    idx = top_indices_conf[0][1]

    return SSDDetection(
        video_id, frame_id,
        x=(det_xmin[idx] + det_xmax[idx]) / 2 * CROP_WIDTH,
        y=(det_ymin[idx] + det_ymax[idx]) / 2 * CROP_HEIGHT,
        w=(det_xmax[idx] - det_xmin[idx]) * CROP_WIDTH,
        h=(det_ymax[idx] - det_ymin[idx]) * CROP_HEIGHT,
        class_id=det_label[idx],
        confidence=det_conf[idx]
    )


class ClassificationDataset(fish_detection.FishDetectionDataset):
    def __init__(self, fold=0, preprocess_input=preprocess_input):
        super().__init__()
        self.preprocess_input = preprocess_input
        print('build clip transforms')
        self.clip_transforms = {
            video_id: self.transform_for_clip(video_id) for video_id in self.video_clips
        }

        self.data = []  # type: List[FishClassification]
        # video_id->frame->species:
        self.known_species = {}  # type: Dict[str, Dict[int, int]]
        self.data, self.known_species = self.load()

        all_video_ids = set(self.video_clips.keys())
        self.test_video_ids = set(dataset.fold_test_video_ids(fold))
        self.train_video_ids = all_video_ids.difference(self.test_video_ids)

        self.train_data = [d for d in self.data if d.video_id in self.train_video_ids]
        self.test_data_full = [d for d in self.data if d.video_id in self.test_video_ids]
        self.test_data = self.test_data_full[::4]

        self.crops_cache = {}

        print('train samples: {} test samples {}'.format(len(self.train_data), len(self.test_data)))

    def train_batches(self, batch_size):
        return int(len(self.train_data) / 2 // batch_size)

    def test_batches(self, batch_size):
        return int(len(self.test_data) // batch_size)

    def load(self):
        repeat_samples = {
            CLASS_FISH_CLEAR_ID: 1,
            CLASS_HAND_OVER_ID: 4,
            CLASS_NO_FISH_ID: 2
        }
        data = []
        known_species = {}
        used_frames = {video_id: set() for video_id in self.video_clips.keys()}
        # we can use the original dataset for clear and no_fish classes
        # ssd detected boxes for extra labeled dataset for all classes, when can guess species from the main dataset
        for video_id, detections in self.detections.items():
            for detection in detections:
                used_frames[video_id].add(detection.frame)

                if detection.class_id != 0:
                    if detection.clip_name not in known_species:
                        known_species[detection.clip_name] = {}
                    known_species[detection.clip_name][detection.frame] = detection.class_id
                    ssd_detection = load_ssd_detection(detection.clip_name, detection.frame)
                    if ssd_detection is not None:
                        data.append(
                            FishClassification(
                                video_id=video_id,
                                frame=detection.frame,
                                x=ssd_detection.x, y=ssd_detection.y, w=ssd_detection.w,
                                species_class=detection.class_id, cover_class=CLASS_FISH_CLEAR_ID
                            )
                        )
                else:
                    data.append(
                        FishClassification(
                            video_id=detection.clip_name,
                            frame=detection.frame,
                            x=0, y=0, w=0, species_class=0, cover_class=CLASS_NO_FISH_ID
                        )
                    )
        print('base size:', len(data))
        # load extra labeled images
        for extra_batch in EXTRA_LABELS_BATCHES:
            for cover_class_id, cover_class in enumerate(COVER_CLASSES):
                for fn in os.listdir(os.path.join(EXTRA_LABELS_BASE_DIR, extra_batch, cover_class)):
                    if not fn.endswith('.jpg'):
                        continue
                    # file name format: video_frame.jpg
                    fn = fn[:-len('.jpg')]
                    video_id, frame = fn.split('_')
                    frame = int(frame) - 1

                    used_frames[video_id].add(frame)

                    for _ in range(repeat_samples[cover_class_id]):
                        if cover_class_id == CLASS_NO_FISH_ID:
                            data.append(
                                FishClassification(
                                    video_id=video_id,
                                    frame=frame,
                                    x=0, y=0, w=0, species_class=0, cover_class=cover_class_id
                                )
                            )
                        else:
                            ssd_detection = load_ssd_detection(video_id, frame)
                            species_class = guess_species(known_species[video_id], frame)
                            if ssd_detection is not None and species_class is not None:
                                data.append(
                                    FishClassification(
                                        video_id=video_id,
                                        frame=frame,
                                        x=ssd_detection.x, y=ssd_detection.y, w=ssd_detection.w,
                                        species_class=species_class, cover_class=cover_class_id
                                    )
                                )

        print('data size:', len(data))
        pickle.dump(used_frames, open('../output/used_frames.pkl', 'wb'))
        return data, known_species

    def generate_xy(self, cfg: SampleCfg):
        img = scipy.misc.imread(dataset.image_crop_fn(cfg.fish_classification.video_id, cfg.fish_classification.frame))

        crop = utils.get_image_crop(full_rgb=img, rect=cfg.rect,
                                    scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
                                    shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
                                    angle=cfg.angle, out_size=INPUT_ROWS)

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.2, r=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, r=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, r=cfg.brightness)

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        return crop * 255.0, cfg.fish_classification.species_class, cfg.fish_classification.cover_class

    def generate(self, batch_size, skip_pp=False, verbose=False):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        while True:
            sample = random.choice(self.train_data)  # type: FishClassification
            cfg = SampleCfg(fish_classification=sample,
                            saturation=rand_or_05(),
                            contrast=rand_or_05(),
                            brightness=rand_or_05(),
                            shift_x_ratio=random.uniform(-0.1, 0.1),
                            shift_y_ratio=random.uniform(-0.1, 0.1),
                            angle=random.uniform(-15.0, 15.0),
                            hflip=random.choice([True, False]),
                            vflip=random.choice([True, False])
                            )
            if verbose:
                print(cfg)
            samples_to_process.append(cfg)

            if len(samples_to_process) == batch_size:
                batch_samples = pool.map(self.generate_xy, samples_to_process)
                # batch_samples = [self.generate_xy(sample) for sample in samples_to_process]
                X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
                y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                if not skip_pp:
                    X_batch = self.preprocess_input(X_batch)
                    y_batch_species = to_categorical(y_batch_species, num_classes=len(SPECIES_CLASSES))
                    y_batch_cover = to_categorical(y_batch_cover, num_classes=len(COVER_CLASSES))
                samples_to_process = []
                yield X_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}

    def generate_test(self, batch_size, skip_pp=False, verbose=False):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        # X_batches = []
        # y_batches = []
        #
        # for i, sample in enumerate(self.test_data[:len(self.test_data) // batch_size * batch_size]):
        #     cfg = SampleCfg(fish_classification=sample)
        #     if verbose:
        #         print(cfg)
        #     samples_to_process.append(cfg)
        #     if i % 1000 == 0:
        #         print(i)
        #
        #     if len(samples_to_process) == batch_size:
        #         batch_samples = pool.map(self.generate_xy, samples_to_process)
        #         X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
        #         y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
        #         y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
        #         if not skip_pp:
        #             X_batch = self.preprocess_input(X_batch)
        #             y_batch_species = to_categorical(y_batch_species, num_classes=len(SPECIES_CLASSES))
        #             y_batch_cover = to_categorical(y_batch_cover, num_classes=len(COVER_CLASSES))
        #         samples_to_process = []
        #         X_batches.append(X_batch)
        #         y_batches.append({'cat_species': y_batch_species, 'cat_cover': y_batch_cover})

        while True:
            # for X_batch, y_batch in zip(X_batches, y_batches):
            #     yield X_batch, y_batch
            for sample in self.test_data[:int(len(self.test_data) / 4 // batch_size) * batch_size]:
                cfg = SampleCfg(fish_classification=sample)
                if verbose:
                    print(cfg)
                samples_to_process.append(cfg)

                if len(samples_to_process) == batch_size:
                    batch_samples = pool.map(self.generate_xy, samples_to_process)
                    # batch_samples = map(self.generate_xy, samples_to_process)
                    X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                    y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
                    y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                    if not skip_pp:
                        X_batch = self.preprocess_input(X_batch)
                        y_batch_species = to_categorical(y_batch_species, num_classes=len(SPECIES_CLASSES))
                        y_batch_cover = to_categorical(y_batch_cover, num_classes=len(COVER_CLASSES))
                    samples_to_process = []
                    yield X_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}


def guess_species(known_species, frame_id):
    known_frames = sorted(known_species.keys())
    if len(known_frames) == 0:
        return None

    for i, frame in enumerate(known_frames):
        if frame == frame_id:
            return known_species[frame]
        elif frame > frame_id:
            if i == 0:
                return known_species[frame]
            if known_species[frame] == known_species[known_frames[i - 1]]:
                return known_species[frame]
            else:
                return None

    return known_species[known_frames[-1]]


def test_guess_species():
    known_species = {2: 1, 5: 1, 7: 2}

    assert guess_species(known_species, 0) == 1
    assert guess_species(known_species, 1) == 1
    assert guess_species(known_species, 2) == 1
    assert guess_species(known_species, 3) == 1
    assert guess_species(known_species, 5) == 1
    assert guess_species(known_species, 6) is None
    assert guess_species(known_species, 7) == 2
    assert guess_species(known_species, 8) == 2


def check_dataset_generator():
    dataset = ClassificationDataset(fold=1)

    batch_size = 4
    for x_batch, y_batch in dataset.generate(batch_size=batch_size, skip_pp=True, verbose=True):
        print(y_batch)
        for i in range(batch_size):
            print(np.min(x_batch[i]), np.max(x_batch[i]))
            plt.imshow(x_batch[i]/256.0)
            # print(SPECIES_CLASSES[y_batch['cat_species'][i]], COVER_CLASSES[y_batch['cat_cover'][i]])
            plt.show()


def train(fold, continue_from_epoch=-1, weights='', batch_size=8):
    dataset = ClassificationDataset(fold=fold)

    model = build_model_densenet_161()
    model.summary()

    model_name = 'model_densenet161'
    checkpoints_dir = '../output/checkpoints/classification/{}_fold_{}'.format(model_name, fold)
    tensorboard_dir = '../output/tensorboard/classification/{}_fold_{}'.format(model_name, fold)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    if len(weights) > 0:
        model.load_weights(weights)

    def cheduler(epoch):
        if epoch < 1:
            return 1e-3
        if epoch < 5:
            return 5e-4
        if epoch < 15:
            return 2e-4
        if epoch < 30:
            return 5e-5
        return 2e-5

    validation_batch_size = 8

    if continue_from_epoch == -1:
        utils.lock_layers_until(model, 'pool5')
        model.summary()
        model.fit_generator(dataset.generate(batch_size=batch_size),
                            steps_per_epoch=dataset.train_batches(batch_size),
                            epochs=1,
                            verbose=1,
                            callbacks=[],
                            validation_data=dataset.generate_test(batch_size=validation_batch_size),
                            validation_steps=dataset.test_batches(validation_batch_size),
                            initial_epoch=continue_from_epoch + 1)
        continue_from_epoch += 1

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=1)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=True)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    utils.lock_layers_until(model, 'pool4')
    model.summary()

    nb_epoch = 400
    model.fit_generator(dataset.generate(batch_size=batch_size),
                        steps_per_epoch=dataset.train_batches(batch_size),
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[
                            checkpoint_periodical,
                            tensorboard,
                            lr_sched
                        ],
                        validation_data=dataset.generate_test(batch_size=validation_batch_size),
                        validation_steps=dataset.test_batches(validation_batch_size),
                        initial_epoch=continue_from_epoch + 1)


def check(fold, weights):
    dataset = ClassificationDataset(fold=fold)

    model = build_model_densenet_161()
    model.load_weights(weights)

    batch_size = 2
    for x_batch, y_batch in dataset.generate(batch_size=batch_size):
        print(y_batch)
        predicted = model.predict_on_batch(x_batch)
        print(predicted)
        for i in range(batch_size):
            plt.imshow(utils.preprocessed_input_to_img_resnet(x_batch[i]))
            true_species = y_batch['cat_species'][i]
            true_cover = y_batch['cat_cover'][i]
            predicted_species = predicted[0][i]
            predicted_cover = predicted[1][i]

            for cls_id, cls in enumerate(SPECIES_CLASSES):
                print('{:12} {:.02f} {:.02f}'.format(cls, true_species[cls_id], predicted_species[cls_id]))

            for cls_id, cls in enumerate(COVER_CLASSES):
                print('{:12} {:.02f} {:.02f}'.format(cls, true_cover[cls_id], predicted_cover[cls_id]))

            print(SPECIES_CLASSES[np.argmax(y_batch['cat_species'][i])],
                  COVER_CLASSES[np.argmax(y_batch['cat_cover'][i])])
            plt.show()


if __name__ == '__main__':
    action = sys.argv[1]
    # action = 'check_dataset_generator'

    # test_guess_species()

    if action == 'train':
        train(fold=int(sys.argv[2]))
    if action == 'check_dataset_generator':
        check_dataset_generator()
    if action == 'check':
        check(fold=int(sys.argv[2]), weights=sys.argv[3])
