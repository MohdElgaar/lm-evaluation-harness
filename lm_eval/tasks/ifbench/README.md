# IFBench

### Paper

IFBench is a benchmark for evaluating precise instruction following in language models. It extends IFEval with 58 new out-of-distribution test constraints covering skills like counting, formatting, and text manipulation. Leading models score below 50% on IFBench while achieving 80%+ on IFEval, demonstrating significant generalization challenges.

Homepage: https://github.com/allenai/IFBench
HuggingFace Dataset: https://huggingface.co/datasets/allenai/IFBench_test

### Citation

```bibtex
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following}, 
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

### Groups and Tasks

#### Tasks

* `ifbench`

### Metrics

* **prompt_level_strict_acc**: Fraction of prompts where all instructions are followed (strict evaluation)
* **inst_level_strict_acc**: Fraction of individual instructions followed (strict evaluation)
* **prompt_level_loose_acc**: Fraction of prompts where all instructions are followed (loose evaluation, allows minor formatting variations)
* **inst_level_loose_acc**: Fraction of individual instructions followed (loose evaluation)

The primary metric reported in the IFBench paper is **prompt_level_loose_acc**.

### Dependencies

Install IFBench task dependencies:

```bash
pip install lm_eval[ifbench]
```

Or install manually: `emoji`, `immutabledict`, `nltk`, `spacy`, `syllapy`

Download the spacy English model:
```bash
python -m spacy download en_core_web_sm
```
