# BERT-of-Theseus
Code for paper ["BERT-of-Theseus: Compressing BERT by Progressive Module Replacing"](http://arxiv.org/abs/2002.02925).

 BERT-of-Theseus is a new compressed BERT by progressively replacing the components of the original BERT.

![BERT of Theseus](https://github.com/JetRunner/BERT-of-Theseus/blob/master/bert-of-theseus.png?raw=true)

## How to run BERT-of-Theseus

### Requirement
Our code is built on [huggingface/transformers](https://github.com/huggingface/transformers). To use our code, you must clone and install [huggingface/transformers](https://github.com/huggingface/transformers).

### Compress a BERT
1. You should fine-tune a predecessor model following the [instruction from huggingface](https://github.com/huggingface/transformers/tree/master/examples#glue) and then save it to a directory if you haven't done so.
2. Run compression following the examples below:
```bash
# For compression with a replacement scheduler
export GLUE_DIR=/path/to/glue_data
export TASK_NAME=MRPC

python ./run_glue.py \
  --model_name_or_path /path/to/saved_predecessor \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 50 \
  --num_train_epochs 50 \
  --output_dir /path/to/save_successor/ \
  --evaluate_during_training \
  --replacing_rate 0.1 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
```

```bash
# For compression with a constant replacing rate
export GLUE_DIR=/path/to/glue_data
export TASK_NAME=MRPC

python ./run_glue.py \
  --model_name_or_path /path/to/saved_predecessor \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 50 \
  --num_train_epochs 50 \
  --output_dir /path/to/save_successor/ \
  --evaluate_during_training \
  --replacing_rate 0.7 \
  --steps_for_replacing 2500 
```
For the detailed description of arguments, please refer to the source code.

## Load Pretrained Model on MNLI

We provide a 6-layer pretrained model on MNLI as a general-purpose model, which can transfer to other sentence classification tasks, outperforming DistillBERT (with the same 6-layer structure) on six tasks of GLUE (dev set).

| Method          | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STS-B |
|-----------------|------|------|------|------|------|-------|-------|
| BERT-base       | 83.5 | 89.5 | 91.2 | 89.8 | 71.1 | 91.5  | 88.9  |
| DistillBERT     | 79.0 | 87.5 | 85.3 | 84.9 | 59.9 | 90.7  | 81.2  |
| BERT-of-Theseus | 82.1 | 87.5 | 88.8 | 88.8 | 70.1 | 91.8  | 87.8  |

You can easily load our general-purpose model using [huggingface/transformers](https://github.com/huggingface/transformers).

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("canwenxu/BERT-of-Theseus-MNLI")

model = AutoModel.from_pretrained("canwenxu/BERT-of-Theseus-MNLI")

```

## Bug Report and Contribution
If you'd like to contribute and add more tasks (only GLUE is available at this moment), please submit a pull request and contact me. Also, if you find any problem or bug, please report with an issue. Thanks!

## Third-Party Implementations
We list some third-party implementations from the community here. Please kindly add your implementation to this list:

`Tensorflow Implementation (tested on NER)`: https://github.com/qiufengyuyi/bert-of-theseus-tf
