echo "CUDA_VISIBLE_DEVICES=0  python -u test.py --log loguvg.txt --testuvg --pretrain Deformable_New_train_vimeo_pengfei1.model --config config.json"
echo "CUDA_VISIBLE_DEVICES=0  python -u test1.py --log loguvg.txt --testuvg --pretrain Deformable_New_train_vimeo_pengfei1.model --config config.json"
echo "CUDA_VISIBLE_DEVICES=0 nohup python -u test.py --log loguvg.txt --testuvg --pretrain Deformable_New_train_vimeo_pengfei1.model --config config.json> test.log 2>&1  &"
echo "CUDA_VISIBLE_DEVICES=0 nohup python -u test1.py --log loguvg.txt --testuvg --pretrain Deformable_New_train_vimeo_pengfei1.model --config config.json> test1.log 2>&1  &"
