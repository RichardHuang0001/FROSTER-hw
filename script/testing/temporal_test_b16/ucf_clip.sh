ROOT=/mnt/SSD8T/home/huangwei/projects/FROSTER
CKPT=$ROOT/checkpoints/basetraining/B2N_hmdb51_froster
OUT_DIR=$CKPT/testing

#加载下载下来的作者在k400上训练好的模型
LOAD_CKPT_FILE=/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_k400_froster/froster_k400_clip_b16.pth

PATCHING_RATIO=0.0

# ucf101_file can be set as ucf101_full, ucf101_split1, ucf101_split2 or ucf101_split3
ucf101_file=ucf101_full
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv

cd $ROOT
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$ucf101_file \
    DATA.PATH_PREFIX $ROOT/data/ucf101/videos \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT//label_rephrase/ucf101_rephrased_classes.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 60 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 101 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
