# 🪶 Quill 

Quill is a WIP attempt at a structurally faithful neural representation of Agda terms, with a current application in premise selection.

---

## 📦 Installation

**1.** Clone or download this repository.

```shell
git clone git@github.com:konstantinosKokos/quill.git
```

**2.** Initialize a fresh python environment (e.g. using [Miniconda](https://docs.anaconda.com/miniconda/)).


 ```shell
conda create --name quill python=3.12
 ``` 

**3.** Activate the environment, and install the package and its dependencies.

```shell
conda activate quill
pip install .
``` 

**Note**: *This will install the cpu version of pytorch by default. For CUDA acceleration, you will need to manually
find and install compatible pytorch and pytorch geometric versions.*

**4.** Download a pretrained model's weights through [here](https://www.dropbox.com/scl/fi/zd2s1evanh7jrrtdofr4d/model.pt?rlkey=yxjj07jvlnfikz1t40e32kguk&st=ccu15xkn&dl=0).

**Note**: *Optional. Only relevant for inference/evaluation.*

**5.** Download the tokenized and processed Agda files from [here](https://www.dropbox.com/scl/fi/bnw4rh6lq5xb7r8j5adpc/tokenized.p?rlkey=ml4h4qpv4n4vrp5c0ysqyus6k&st=neg9zynu&dl=0).  

**Note**: *Optional. Only relevant for training/evaluation.*

You can **uninstall** by removing the environment from your system, and deleting all downloaded files.

---

## ❓ How-To

### 🪿 Interface with Agda

> 🚧 **Note**: Work in Progress 🚧
>
> Bear in mind that the main branches of agda2train and quill are not necessarily in sync.
 
You should be able to export an Agda file (possibly containing holes) into json format 
(see [agda2train](https://github.com/omelkonian/agda2train)), and then invoke a script that looks like `/scripts/inference.py`
to obtain lemma suggestions.
 

The second part of this process is partially streamlined through a CLI API.

First, deploy a server running the model in inference mode:

```shell
agda-quill serve -config PATH_TO_MODEL_CONFIG -weights PATH_TO_MODEL_WEIGHTS
```

You can then optionally precompute representations of the lemmas defined in various files/libraries:

```shell
agda-quill cache -files ./data/stdlib/Data.List.*
```

Finally, you can query the model for suggestions, optionally using the cached representations:
```shell
agda-quill query -file ./data/stdlib/Algebra.Construct.NaturalChoice.Min.json --max_suggestions 2 --use_cache
```

### 🤖 Train a model ...

#### ... on the original train/dev split 
Adjust the training configuration file (`/data/config.json`) and run the training script (`/scripts/train.py`), after any necessary modifications.

#### ... on a new dataset

> ⚠️ **Warning**: Unpinned Dependencies ⚠️
>
> The current packaging of quill is tied to the bundled `tokenized.p`. Differences in agda2train versions might necessitate different preprocessing or data digestion pipelines. 

Gather a bunch of JSON exports over an Agda library of your choice (again, using [agda2train](https://github.com/omelkonian/agda2train)).
Run the preprocessing script (`/scripts/preprocess.py`) to have Python parse and tokenize the exports.
Then train as you would normally.

### 📈 Evaluate a model
Run the evaluation script (`/scripts/validate.py`), after any necessary modifications. 

### 🔎 Inspect the parsed & tokenized files
The Python definitions of Agda file- and term-structures can be found in `src/quill/data/agda/syntax.py`.
The definitions of tokenized files can be found in `/src/quill/data/tokenization.py`. 
Use them as you see fit to navigate and process the parsed & tokenized files. 
You can load these with:
```python3
import pickle

with open('./data/tokenized_sample.p', 'rb') as f:
    samples = pickle.load(f)
```

### 🖼️ Visualize lemma representations
For a low-dimensional projection of the neural representations of (most) lemmas in the stdlib,
together with their undirectional reference structure, look here: https://konstantinoskokos.github.io/quill/viz.html.

### 💬 Comment, ask, complain, etc.
Feel free to open and issue or get in touch.

---

## 📜 Citing
You can cite the following entry if using our work in  a scholarly context.

```bibtex
@inproceedings{kogkalidis_learning_2024,
    author = {Konstantinos Kogkalidis and Orestis Melkonian and Jean-Philippe Bernardy},
    title = {Learning Structure-Aware Representations of Dependent Types},
    booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year = {2024},
}
```
