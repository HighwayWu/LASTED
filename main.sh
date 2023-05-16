python main.py \
    --model 'LASTED' \
    --train_file 'annotation/Train_num398700.txt' \
    --num_class 4 \
    --val_ratio 0.005 \
    --test_file 'annotation/Test_MidjourneyV5_num2000.txt' \
    --isTrain 0 \
    --lr 0.0001 \
    --resume 'weights/LASTED_pretrained.pt' \
    --data_size 448 \
    --batch_size 48 \
    --gpu '0,1,2,3' \
    2>&1 | tee weights/log.log
