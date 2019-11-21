#/************************************************************************************
#***
#***	Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2019-11-05 16:20:01
#***
#************************************************************************************/
#
#! /bin/sh

usage()
{
	echo "Usage: $0 input_video_name output_video_name"

	exit 1
}

predict_b()
{
	CHECKPOINT_FILE=output/WDSR-B-f32-b16-r6-x4-best.pth.tar
	python predict.py  \
        --checkpoint-file ${CHECKPOINT_FILE} \
        --scale 4 \
        --n-feats 32 \
        --n-res-blocks 16 \
        --model "WDSR-B" \
        --expansion-ratio 6 \
        --input $1 \
        --output /tmp/sdr
}


[ "$1" = "" ] && usage
[ "$2" = "" ] && usage


predict_b $1 /tmp/sdr
video_coder --encode /tmp/sdr/%3d.png $2
# rm -rf /tmp/sdr


