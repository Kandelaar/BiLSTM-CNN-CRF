START_TAG = '<START>'
STOP_TAG = '<STOP>'
PAD_TAG = '<PAD>'
pad_id = 0
batch_size = 10
clipping = 5.0

idx2tag = [PAD_TAG, "O", "I-MISC", "I-ORG", "I-LOC", "I-PER", "B-MISC", "B-ORG", "B-LOC", "B-PER",
           START_TAG, STOP_TAG]
tag2idx = {v: i for i, v in enumerate(idx2tag)}

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
