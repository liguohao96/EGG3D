
python dataset_preprocessing_egg3d/ffhq/run_faceparsing.py \
    --indir  ${DATA_ROOT}/final_crops \
    --outdir ${DATA_ROOT}/faceparsing

python dataset_preprocessing_egg3d/ffhq/run_emoca_model.py \
    --indir  ${DATA_ROOT}/final_crops \
    --outdir ${DATA_ROOT}/emocaresult

python dataset_preprocessing_egg3d/ffhq/run_photometric_v4.py \
    --indir  ${DATA_ROOT}/final_crops \
    --fpdir  ${DATA_ROOT}/faceparsing \
    --frdir  ${DATA_ROOT}/emocaresult \
    --fitdir ${DATA_ROOT}/photometric_v4 \
    --segdir ${DATA_ROOT}/segment_img_v4

python dataset_preprocessing_egg3d/ffhq/pack_all.py \
    --indir   ${DATA_ROOT}/final_crops \
    --segdir  ${DATA_ROOT}/segment_img_v4 \
    --fitdir  ${DATA_ROOT}/photometric_v4 \
    --outfile Data/FFHQ_ex3d_v4.zip
