DIV2K_DIR=/home/dell/ZDisk/WorkSpace/4K/dataset/DIV2K
CHECKPOINT_FILE=output/WDSR-B-f32-b16-r6-x2-best.pth.tar

python eval.py --dataset-dir ${DIV2K_DIR} \
                --checkpoint-file ${CHECKPOINT_FILE} \
                --model "WDSR-B" \
                --scale 2 \
                --n-feats 32 \
                --n-res-blocks 16 \
                --expansion-ratio 6 \
                --low-rank-ratio 0.8 \
                --res-scale 1.0 \
