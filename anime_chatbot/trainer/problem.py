"""
Generate an anime subtitle response for an anime subtitle input sentence.
"""

import os
import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems

tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file


@registry.register_problem
class AnimeChatbotProblem(text_problems.Text2TextProblem):
  """Generate an anime subtitle response for an anime subtitle input sentence."""

  @property
  def approx_vocab_size(self):
    return 2 ** 16  # ~64k

  @property
  def is_generate_per_split(self):
    # generate_data will NOT shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
      "split": problem.DatasetSplit.TRAIN,
      "shards": 90,
    }, {
      "split": problem.DatasetSplit.EVAL,
      "shards": 10,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(cwd + '/../../input.txt', 'r') as input_file:
      with open(cwd + '/../../output.txt', 'r') as output_file:
        input_line = input_file.readline()
        output_line = output_file.readline()
        while input_line and output_line:
          yield {
            "inputs": input_line,
            "targets": output_line
          }
          input_line = input_file.readline()
          output_line = output_file.readline()


# Smaller than the typical translate model, and with more regularization
@registry.register_hparams
def transformer_anime_chatbot():
  hparams = transformer.transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  hparams.attention_dropout = 0.6
  hparams.layer_prepostprocess_dropout = 0.6
  hparams.learning_rate = 0.05
  return hparams


@registry.register_hparams
def transformer_anime_chatbot_tpu():
  hparams = transformer_anime_chatbot()
  transformer.update_hparams_for_tpu(hparams)
  return hparams


# hyperparameter tuning ranges
@registry.register_ranged_hparams
def transformer_anime_chatbot_range(rhp):
  rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
  rhp.set_int("num_hidden_layers", 2, 8)
  rhp.set_discrete("hidden_size", [128, 256, 512])
  rhp.set_float("attention_dropout", 0.4, 0.7)
  rhp.set_discrete("num_heads", [2, 4, 8, 16, 32, 64, 128])
  rhp.set_discrete("filter_size", [512, 1024])
