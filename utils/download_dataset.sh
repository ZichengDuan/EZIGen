export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
--repo-type model \
--resume-download facebook/sam2-hiera-large \
--local-dir /mnt_det2/muqi.db/workdir/pretrained_models/lotus-normal-g-v1-1 \
--local-dir-use-symlinks False \
--token ***REMOVED*** 