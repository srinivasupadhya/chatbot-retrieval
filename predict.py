import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

tf.flags.DEFINE_string("saved_model_dir", None, "Directory to load saved model from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.saved_model_dir:
  print("You must specify a model directory")
  sys.exit(1)

INPUT_FEATURE_SIZE = 160

start_time = time.time()

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_processor_file)

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": np.reshape(context_matrix, (1,INPUT_FEATURE_SIZE)),
    "context_len": np.reshape(context_len, (1,1)),
    "utterance": np.reshape(utterance_matrix, (1,INPUT_FEATURE_SIZE)),
    "utterance_len": np.reshape(utterance_len, (1,1)),
  }
  return features

INPUT_CONTEXT = "my monitor is not working"
POTENTIAL_RESPONSES = ["Welcome Is there anything else I could help you with", "What is your OS version",
                       "Good Morning What is the problem you are facing", "Looking into the issue",
                       "Hi How can I help you today", "what is the model number", "have you tried upgrading the driver"]

predict_fn = predictor.from_saved_model(FLAGS.saved_model_dir)
result = []
for r in POTENTIAL_RESPONSES:
    prediction = predict_fn(get_features(INPUT_CONTEXT, r.lower()))['output']
    result.append((r.lower(), prediction))

print(sorted(result, key=lambda x: x[1], reverse=True), sep='\n')
print("--- %s seconds ---" % (time.time() - start_time))