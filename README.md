# SPaCE
Experiments for the BIG@ICML 2020 workshop paper  "Towards Self-Paced Context Evaluations for Contextual Reinforcement Learning"

    @inproceedings{eimer-bigicml20,
      author    = {T. Eimer and A. Biedenkapp and F. Hutter and M. Lindauer},
      title     = {Towards Self-Paced Context Evaluations for Contextual Reinforcement Learning},
      booktitle = {Workshop on Inductive Biases, Invariances and Generalization in {RL} ({BIG@ICML}'20)},
      year = {2020},
      month     = jul,
    }

## Setup & Usage
To run the experiments, you need to install the dependencies:
```bash
pip install -r requirements
```
The included notebooks for plotting and generating new instance also require jupyter to be installed:
```bash
pip install jupyter
```

To train a SPaCE agent on our provided instances, run:
```bash
python src/ray_spl.py --mode spl --instances features/cpm_train.csv --test features/cpm_test.csv
```
Replacing "spl" with "rr" will result in a round robin trained agent for comparison.

## Results included in the workshop paper
Our own training results are included in this repository. To plot the data, you can use the provided jupyter notebook.
