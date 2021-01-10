# Convert to Hugging Face compatible checkpoint
# python convert_to_hf_ckpt.py /path/to/model_to_convert /path/to/dir_to_save

from bert_of_theseus.modeling_bert_of_theseus import BertForSequenceClassification
import sys
import os

model = BertForSequenceClassification.from_pretrained(sys.argv[1])
model.bert.encoder.layer = model.bert.encoder.scc_layer
model.bert.config.num_hidden_layers = model.bert.encoder.scc_n_layer
del model.bert.encoder.scc_layer
if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])
model.save_pretrained(sys.argv[2])
