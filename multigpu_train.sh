python multigpu_train.py \
	--gpu_list=0,1 \
	--input_size=512 \
	--batch_size_per_gpu=16 \
	--checkpoint_path=./checkpoints/ \
	--text_scale=512 \
	--training_data_path=../../Dataset/MLT2019/MLT_train+validation \
	--geometry=RBOX \
	--learning_rate=0.0002 \
	--num_readers=24 \
        --restore = True
	#--pretrained_model_path=./base_model/model.ckpt-516002 	# Here we don't adopt a pre-trained model. If you want to use the pre-trained model, you need download it yourself (Please refer to https://github.com/argman/EAST)
