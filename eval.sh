DIV2K_DIR=/home/dell/ZDisk/WorkSpace/4K/dataset/DIV2K


test_a()
{
	CHECKPOINT_FILE=output/WDSR-A-f32-b16-r4-x2-best.pth.tar
	python eval.py --dataset-dir ${DIV2K_DIR} \
	                --checkpoint-file ${CHECKPOINT_FILE} \
	                --model "WDSR-A" \
	                --expansion-ratio 4
}

test_b()
{
	CHECKPOINT_FILE=output/WDSR-B-f32-b16-r6-x4-best.pth.tar
	python eval.py --dataset-dir ${DIV2K_DIR} \
	                --checkpoint-file ${CHECKPOINT_FILE} \
	                --scale 4 \
	                --n-feats 32 \
	                --n-res-blocks 16 \
	                --model "WDSR-B" \
	                --expansion-ratio 6
}

test_b
