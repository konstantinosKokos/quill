# 🪶 Quill 

Quill is a WIP attempt at a structurally faithful neural representation of Agda terms, with an application in premise selection.

## How-To

### Interface with Agda

TBD / Help needed. 
Right now you can export a file (possibly containing holes) into json format
(see https://github.com/omelkonian/agda2train), and then invoke a script that looks like `/scripts/inference.py`
to obtain lemma suggestions.

### Train a model

#### On the original train/dev split
Simply download the binary dump containing processed and tokenized agda files ([link](https://www.dropbox.com/scl/fi/bnw4rh6lq5xb7r8j5adpc/tokenized.p?rlkey=ml4h4qpv4n4vrp5c0ysqyus6k&st=neg9zynu&dl=0)).
Adjust the training configuration file (`/data/config.json`) if/as needed. 
Then run the training script (`/scripts/train.py`), after any necessary modifications.

### On a new dataset
Gather a bunch of JSON exports over an Agda library of your choice using [agda2train](https://github.com/omelkonian/agda2train).
Run the preprocessing script (`/scripts/preprocess.py`) to have Python parse and tokenize the exports.
Then train as you would normally.

### Evaluate a model
Run the evaluation script (`/scripts/validate.py`), after any necessary modifications. You can download one of the pretrained models [here](https://www.dropbox.com/scl/fi/58i2mhpfkctp9lasw3fc6/model.pt?rlkey=8dhc69p9798r9drskcx06j449&st=talhqkgl&dl=0).

### Inspect the parsed & tokenized files
The Python definitions of Agda file- and term-structures can be found in `src/quill/data/agda/syntax.py`.
The definitions of tokenized files can be found in `/src/quill/data/tokenization.py`. 
Use them as you see fit to navigate and process the parsed & tokenized files. 
You can load these with:
```python3
import pickle

with open('./data/tokenized_sample.p', 'rb') as f:
    samples = pickle.load(f)
```
If you want to work on the full output (rather then the minimal sample contained in the repo), download the file from [here](https://www.dropbox.com/scl/fi/bnw4rh6lq5xb7r8j5adpc/tokenized.p?rlkey=ml4h4qpv4n4vrp5c0ysqyus6k&st=neg9zynu&dl=0).

### Comment / Ask / Complain / ...
Feel free to open and issue or get in touch.

## Lemma Visualization
For a low-dimensional projection of the neural representations of (most) lemmas in the stdlib,
together with their undirectional reference structure, look here: https://konstantinoskokos.github.io/quill/viz.html.


## Citing
You can cite the following entry if using our work in  a scholarly context.

```bibtex
@inproceedings{kogkalidis_learning_2024,
    author = {Konstantinos Kogkalidis and Orestis Melkonian and Jean-Philippe Bernardy},
    title = {Learning Structure-Aware Representations of Dependent Types},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year = {2024},
}
```
