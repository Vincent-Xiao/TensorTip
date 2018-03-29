#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download and preprocess ImageNet Challenge 2012
# training and validation data set.
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_imagenet_data.py for
# details of how the Example protocol buffers contain the ImageNet data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-01023-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
#
# Note that this script may take several hours to run to completion. The
# conversion of the ImageNet data to TFRecords alone takes 2-3 hours depending
# on the speed of your machine. Please be patient.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  ./download_and_preprocess_imagenet.sh [data-dir] [ifdownload:download or process]
set -e

if [ -z "$1" ]; then
  echo "Usage: download_and_preprocess_imagenet.sh [data dir] [ifdownload]"
  exit
fi
if [ -z "$2" ]; then
  echo "Usage: download_and_preprocess_imagenet.sh [data dir] [ifdownload]"
  exit
fi

# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${SCRATCH_DIR}"
WORK_DIR=""

# Download the ImageNet data.
LABELS_FILE="${WORK_DIR}imagenet_lsvrc_2015_synsets.txt"
if [ $2 != "process" ]; then
    echo "Downloading ImageNet Dataset"
    DOWNLOAD_SCRIPT="${WORK_DIR}download_imagenet.sh"
    bash "${DOWNLOAD_SCRIPT}" "${SCRATCH_DIR}" "${LABELS_FILE}"
fi
# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}/ILSVRC2012_img_train/"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/ILSVRC2012_img_val/"

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="${WORK_DIR}preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="${WORK_DIR}imagenet_2012_validation_synset_labels.txt"
python "${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="${WORK_DIR}process_bounding_boxes.py"
BOUNDING_BOX_FILE="${SCRATCH_DIR}/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${SCRATCH_DIR}/ILSVRC2012_bbox_train/"

python "${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
| sort > "${BOUNDING_BOX_FILE}"
echo "Finished downloading and preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="${WORK_DIR}build_imagenet_data.py"
OUTPUT_DIRECTORY="${DATA_DIR}/TFRecords"
IMAGENET_METADATA_FILE="${WORK_DIR}imagenet_metadata.txt"

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"
