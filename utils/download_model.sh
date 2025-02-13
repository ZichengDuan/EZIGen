export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
--repo-type model \
--resume-download stabilityai/stable-diffusion-2-1-base \
--local-dir /mnt/sh_nas/duanzicheng.dzc/hf_downloads/models/stabilityai--stable-diffusion-2-1-base \
--local-dir-use-symlinks False \
--token ***REMOVED*** 
