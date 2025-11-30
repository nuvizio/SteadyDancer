uv run python generate_dancer.py \
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir checkpoints/Wan2.1-T2V-14B \
    --prompt "Two anthropomorphic cats in comfy clothes cooking a romantic dinner, high quality, 4k, photorealistic" \
    --image data/images/00001.png \
    --cond_pos_folder preprocess/output/video00001_img00001/example/positive \
    --cond_neg_folder preprocess/output/video00001_img00001/example/negative \
    --save_file test_i2v_dance_video.mp4

