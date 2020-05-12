import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

INPUT_FEATURE_SIZE = 160

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

predict_fn = predictor.from_saved_model("./saved_model/1590301632/1590301684")
result = []
for r in POTENTIAL_RESPONSES:
    prediction = predict_fn(get_features(INPUT_CONTEXT, r.lower()))['output']
    print(r.lower(), prediction)
