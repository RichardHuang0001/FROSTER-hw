ROOT=/mnt/SSD8T/home/huangwei/projects/FROSTER
CKPT=$ROOT/checkpoints/basetraining/B2N_hmdb51_froster
OUT_DIR=$CKPT/testing
LOAD_CKPT_FILE=/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_k400_froster/froster_k400_clip_b16.pth

hmdb_file=hmdb_full
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv

rephrased_file=test_rephrased.json 
# hmdb_file can be set as hmdb_full, hmdb_split1, hmdb_split2, hmdb_split3

cd $ROOT
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$hmdb_file \
    DATA.PATH_PREFIX $ROOT/data/hmdb51 \
    DATA.PATH_LABEL_SEPARATOR , \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_rephrase/hmdb_rephrased_classes.json  \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 60 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 51 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt  \
