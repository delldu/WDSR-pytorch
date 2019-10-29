DIV2K_DIR=/home/dell/ZDisk/WorkSpace/4K/dataset/DIV2K
OUTPUT_DIR=output

train_b2()
{
	python train.py --dataset-dir ${DIV2K_DIR} \
	                --output-dir ${OUTPUT_DIR} \
	                --model "WDSR-B" \
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
	python train.py --dataset-dir ${DIV2K_DIR} \
	                --output-dir ${OUTPUT_DIR} \
	                --model "WDSR-B" \
	                --scale 4 \
	                --n-feats 32 \
	                --n-res-blocks 16 \
	                --expansion-ratio 6 \
	                --low-rank-ratio 0.8 \
	                --res-scale 1.0 \
	                --lr 1e-3
}

train_b4
