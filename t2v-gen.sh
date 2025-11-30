uv run python generate_dancer.py \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir checkpoints/Wan2.1-T2V-14B \
    --prompt "Two anthropomorphic cats in comfy clothes cooking a romantic dinner, high quality, 4k, photorealistic" \
    --save_file test_t2v_firstframe_dance_video.mp4

