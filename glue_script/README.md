# GLUE Submission Script

This script (by [@MichaelZhouwang](https://github.com/MichaelZhouwang)) is for predicting with [huggingface/transformers](https://github.com/huggingface/transformers) on GLUE test set. This script is adapted from the original `run_glue.py`.

It helps you prepare your submission to [GLUE Leaderboard](https://gluebenchmark.com/).

If you find this script helpful, please kindly consider citing our paper:

```bibtex
@misc{xu2020bertoftheseus,
    title={BERT-of-Theseus: Compressing BERT by Progressive Module Replacing},
    author={Canwen Xu and Wangchunshu Zhou and Tao Ge and Furu Wei and Ming Zhou},
    year={2020},
    eprint={2002.02925},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Usage
Please first replace the `glue.py` in `src/transformers/data/processor/` by ours and then use `run_prediction.py` in the same way of `run_glue.py`.
