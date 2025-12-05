python test.py \
  --dataset mvtec \
  --data_path ./data/mvtec \
  --ckpt_path ./filo_train_on_mvtec.pth \
  --grounded_checkpoint ./grounding_train_on_mvtec.pth \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --area_threshold 0.7 \
  --device cuda
