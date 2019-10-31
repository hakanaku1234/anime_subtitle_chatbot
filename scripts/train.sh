t2t-datagen \
--t2t_usr_dir=anime_chatbot/trainer \
--problem=anime_chatbot_problem \
--data_dir=data/output/t2t_data \
--tmp_dir=data/output/t2t_data/tmp


nohup \
t2t-trainer \
  --data_dir=gs://bucket/anime_chatbot/data \
  --t2t_usr_dir=anime_chatbot/trainer \
  --problem=anime_chatbot_problem \
  --model=transformer \
  --output_dir=gs://bucket/anime_chatbot/model_tpu \
  --train_steps=100000000 \
  --use_tpu=True \
  --cloud_tpu_name=node-1 \
  --hparams_set=transformer_anime_chatbot_tpu \
  --eval_steps=10 \
&
