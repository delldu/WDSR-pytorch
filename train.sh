DIV2K_DIR=/home/dell/ZDisk/WorkSpace/4K/dataset/DIV2K
OUTPUT_DIR=output

train_b2()
{
	CHECKPOINT_FILE=output/WDSR-A-f32-b16-r4-x2-best.pth.tar

	python train.py --dataset-dir ${DIV2K_DIR} \
	                --output-dir ${OUTPUT_DIR} \
	                --model "WDSR-B" \
           	        --checkpoint-file ${CHECKPOINT_FILE} \
	                --scale 2 \
	                --n-feats 32 \
	                --n-res-blocks 16 \
	                --expansion-ratio 6 \
	                --low-rank-ratio 0.8 \
	                --res-scale 1.0 \
	                --lr 1e-3
}

train_b4()
{
	CHECKPOINT_FILE=output/WDSR-B-f32-b16-r6-x4-best.pth.tar

	python train.py --dataset-dir ${DIV2K_DIR} \
	                --output-dir ${OUTPUT_DIR} \
	                --model "WDSR-B" \
	                --checkpoint-file ${CHECKPOINT_FILE} \
	                --scale 4 \
	                --n-feats 32 \
	                --n-res-blocks 16 \
	                --expansion-ratio 6 \
	                --low-rank-ratio 0.8 \
	                --res-scale 1.0 \
	                --lr 1e-4 \
	                --epochs 300
}

train_b4

