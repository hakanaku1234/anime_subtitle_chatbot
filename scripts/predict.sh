# batch decoding on TPU (good for testing purposes)
t2t-decoder \
  --data_dir=gs://bucket/anime_chatbot/data \
  --problem=anime_chatbot_problem \
  --model=transformer \
  --hparams_set=transformer_anime_chatbot_tpu \
  --output_dir=gs://bucket/anime_chatbot/model_tpu \
  --t2t_usr_dir=anime_chatbot/trainer \
  --decode_hparams="beam_size=4,alpha=0.6" \
  --decode_from_file=phrases_input.txt \
  --use_tpu=True \
  --cloud_tpu_name=node-1

# 0.5 second inference on CPU
t2t-decoder \
  --data_dir=gs://bucket/anime_chatbot/data \
  --problem=anime_chatbot_problem \
  --model=transformer \
  --hparams_set=transformer_anime_chatbot_tpu \
  --output_dir=gs://bucket/anime_chatbot/model_tpu \
  --t2t_usr_dir=anime_chatbot/trainer \
  --decode_hparams="beam_size=4,alpha=0.6" \
  --decode_interactive