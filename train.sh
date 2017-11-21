#!/usr/bin/env bash

set -x

# train UNET based segmentation model for ruler mask detection
python3 ruler_masks.py train
python3 ruler_masks.py predict
python3 ruler_masks.py find_ruler_angles
python3 ruler_masks.py find_ruler_vectors
# after this stage, information about ruler points is generated at ../output/ruler_points.csv

python3 ruler_masks.py generate_crops
# after this stage, rotated crops around ruler are generated at ../output/ruler_crops
# worth to check the ruler is centered horisontaly and occupies space on from approx x = w*0.2 to w*0.8


# train the fish detection model:
python3 fish_detection_ssd.py train_resnet

# resnet_53 - more covered crops by hands, useful for next stage training
# resnet_62 - less covered crops, may be useful for sequence predicting

# I selected checkpoints 23 and 45 blindly here, usually I checked the behavior of the model
# on frames not used for training

# last 2 arguments: from_idx and count. May be used to run prediction in parallel on multiple gpu
# for example on one gpu:
# python3 fish_detection_ssd.py ... 0 700
# on another
# python3 fish_detection_ssd.py ... 700 1400
python3 fish_detection_ssd.py generate_predictions_on_train_clips ../output/checkpoints/detect_ssd/ssd_resnet_720/checkpoint-023*.hdf5 resnet_53 0 1400
python3 fish_detection_ssd.py generate_predictions_on_train_clips ../output/checkpoints/detect_ssd/ssd_resnet_720/checkpoint-045*.hdf5 resnet_62 0 1400
# after this stage, ../output/predictions_ssd_roi2/resnet_53 and ../output/predictions_ssd_roi2/resnet_62 are generated

python3 fish_classification.py save_detection_results --detection_model resnet_53
python3 fish_classification.py save_detection_results --detection_model resnet_62

# generate crops with fish used by classification networks
python3 fish_classification.py generate_train_classification_crops --detection_model resnet_53
python3 fish_classification.py generate_train_classification_crops --detection_model resnet_62

# train densenet 161 classification model
python3 fish_classification.py train --fold 1 --classification_model densenet
python3 fish_classification.py train --fold 2 --classification_model densenet
python3 fish_classification.py train --fold 3 --classification_model densenet
python3 fish_classification.py train --fold 4 --classification_model densenet

# let's choose checkpoint 006 for densenet
for fold in 1 2 3 4
do
    pushd ../output/checkpoints/classification/model_densenet161_ds3_fold_${fold}
    ln -s checkpoint-006-*.hdf5 checkpoint-selected.hdf5
    popd

    python3 fish_classification.py generate_results_from_detection_crops_on_fold --fold ${fold} --detection_model resnet_62 --classification_model densenet --weights ../output/checkpoints/classification/model_densenet161_ds3_fold_${fold}/checkpoint-selected.hdf5
    python3 fish_classification.py generate_results_from_detection_crops_on_fold --fold ${fold} --detection_model resnet_53 --classification_model densenet --weights ../output/checkpoints/classification/model_densenet161_ds3_fold_${fold}/checkpoint-selected.hdf5
done

python3 fish_classification.py combine_results --detection_model resnet_53 --classification_model densenet
python3 fish_classification.py combine_results --detection_model resnet_62 --classification_model densenet

# train RNN based key frames sequence model:

python3 fish_sequence_rnn.py train_full

# train inception v3 classification model
for fold in 1 2 3 4
do
    python3 fish_classification.py train --fold ${fold} --classification_model inception

    # let's choose checkpoint 004 for inception v3 model, to be used by prediction stage
    pushd ../output/checkpoints/classification/model_inception_fold_${fold}
    ln -s checkpoint-004-*.hdf5 checkpoint-selected.hdf5
    popd
done
