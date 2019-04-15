python eval.py \
	--test_data_path=../../Dataset/MLT2019/ICDAR2015_test_img  \
	--gpu_list=2 \
	--checkpoint_path=./checkpoints_train_MLT2017/ \
	--output_dir=./output/train_MLT_test_ICDAR2015 \
        --no_write_images=True \
        --confidence=False

