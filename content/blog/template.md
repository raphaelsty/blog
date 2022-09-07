+++
date = "2022-09-06"
draft = false
title = "Extractive Question Answering application."
+++

This blog post is a response to Max Halford's blog post ["NLP at Carbonfact: how would you do it?](https://maxhalford.github.io/blog/carbonfact-nlp-open-problem/)

In his blog post, Max proposes a procedure to automate a part of clothes' life cycle analysis (LCA). The LCA aims, among other things, to estimate the amount of carbon needed to produce a good. 

The specific task he is working to solve is to identify named entities and create structured data from those entities. But here's the thing: Max's clients' data all have distinct formats, so much so that they agreed to have to write custom data normalization logic for each of their clients.

Sample data is available for [download](https://maxhalford.github.io/files/datasets/nlp-carbonfact/inputs.txt); this is what they look like:

```
lace 87% nylon 13% spandex; mesh: 95% nylon 5% spandex
```

Max's objective is to extract the structured information associated with these descriptions. He manually annotated the above data to facilitate information extraction via a machine learning algorithm. The ground truth is also available for [download](https://maxhalford.github.io/files/datasets/nlp-carbonfact/outputs.json).

```json
{
    "lace": [
        {
            "material": "nylon",
            "proportion": 87.0
        },
        {
            "material": "spandex",
            "proportion": 13.0
        }
    ],
    "mesh": [
        {
            "material": "nylon",
            "proportion": 95.0
        },
        {
            "material": "spandex",
            "proportion": 5.0
        }
    ]
}
```

In the context of this response, my objective is to propose a model capable of adapting to different formats to facilitate the annotation of new data and to define a baseline for future improvement. My solution comprises an extractive question-answering model dedicated to entity recognition and an information retrieval pipeline for their disambiguation. 

<details>
<summary>Click here to see the version of the Python packages I use in the tutorial.</summary>

Transformer training packages:

```python
!pip install transformers==4.17.0
!pip install datasets==2.0.0
```

Packages dedicated to inference:

```pyhon
!pip install cherche==0.8.0
!pip install transformers==4.17.0
!pip install spacy==3.3.0
!python -m spacy download en_core_web_sm
```

</details>

## Question-answering dataset

Let's start by creating a dataset from the annotated data to train a question-answering model. 

For each item, we can create sample questions to identify:

- Components: `What is the {i-th} component ?`
- Materials and proportions: `What is the {i-th} material of the component {component} ?`

```
body: 83% nylon 17% spandex; lace: 86% nylon 14% spandex
```

The following are examples of questions on components.

```json
[
    {
        "answers": {"answer_start": [7], "text": ["body"]},
        "context": "[NONE] body 83 nylon 17 spandex lace 86 nylon 14 spandex",
        "question": "What is the 1 component ?",
    },
    {
        "answers": {"answer_start": [32], "text": ["lace"]},
        "context": "[NONE] body 83 nylon 17 spandex lace 86 nylon 14 spandex",
        "question": "What is the 2 component ?",
    },
]
```

Some examples do not have components specified, so we concatenate the `[NONE]` tag in front of all the documents in the corpus and ask the model to mention the tag when this is the case. 

The following are examples of questions for materials and proportions. We combine proportions and materials because they follow each other systematically in the training data. Furthermore, we will have to make a single call to our model to extract both pieces of information, saving us computing power.

```json
[
    {
        "answers": {"answer_start": [12], "text": ["83 nylon"]},
        "context": "[NONE] body 83 nylon 17 spandex lace 86 nylon 14 spandex",
        "question": "What is the 1 material of the component body ?",
    },
    {
        "answers": {"answer_start": [21], "text": ["17 spandex"]},
        "context": "[NONE] body 83 nylon 17 spandex lace 86 nylon 14 spandex",
        "question": "What is the 2 material of the component body ?",
    },
    {
        "answers": {"answer_start": [37], "text": ["86 nylon"]},
        "context": "[NONE] body 83 nylon 17 spandex lace 86 nylon 14 spandex",
        "question": "What is the 1 material of the component lace ?",
    },
    {
        "answers": {"answer_start": [46], "text": ["14 spandex"]},
        "context": "[NONE] body 83 nylon 17 spandex lace 86 nylon 14 spandex",
        "question": "What is the 2 material of the component lace ?",
    },
]
```

**We create 2639 questions from 600 descriptions or an average of 4.39 questions per document. Then, we divide these questions into 2139 training questions and 500 testing questions dedicated to evaluating our model.**

Here is the code to create the train and test datasets:

<details>
<summary>Click to see the code.</summary>

```python
import json
import re


# Keeps some of the pre-processing to increase the amount of training data.
def replace(text):
    text = text.replace("top body", "top_body")
    text = text.replace("op body", "top_body")
    text = text.replace("body & panty", "body_panty")
    text = text.replace("edge lace", "edge_lace")
    text = text.replace("edg lace", "edge_lace")
    text = text.replace("cup shell", "cup_shell")
    text = text.replace("centre front and wings", "centre_front_and_wings")
    text = text.replace("cup lining", "cup_lining")
    text = text.replace("front panel", "front_panel")
    text = text.replace("back panel", "back_panel")
    text = text.replace("marl fabric", "marl_fabric")
    text = text.replace("knited top", "knitted_top")
    text = text.replace("striped mesh", "striped_mesh")
    text = text.replace("trim lace", "trim_lace")
    text = text.replace("trim lace", "trim_lace")

    # typos
    text = text.replace("sapndex", "spandex")
    text = text.replace("spadnex", "spandex")
    text = text.replace("spandexndex", "spandex")
    text = re.sub("span$", "spandex", text)
    text = re.sub("spande$", "spandex", text)
    text = text.replace("polyest ", "polyester ")
    text = re.sub("polyeste$", "polyester", text)
    text = re.sub("poly$", "polyester", text)
    text = text.replace("polyster", "polyester")
    text = text.replace("polyeste ", "polyester ")
    text = text.replace("elastanee", "elastane")
    text = text.replace(" poly ", " polyester ")
    text = text.replace("cotton algodón coton", "cotton")
    text = text.replace("poliamide", "polyamide")
    text = text.replace("recycle polyamide", "recycled polyamide")
    text = text.replace("polyester poliéster", "polyester")
    text = text.replace("polystester", "polyester")
    text = text.replace("regualar polyamide", "regular polyamide")
    text = text.replace("recycle nylon", "recycled nylon")
    text = text.replace("buttom", "bottom")
    text = text.replace("recycle polyester", "recycled polyester")
    text = text.replace("125", "12%")
    text = text.replace("135", "13%")
    text = text.replace("recycled polyeser", "recycled polyester")
    text = text.replace("polyeter", "polyester")
    text = text.replace("polyeseter", "polyester")
    text = text.replace("viscouse", "viscose")
    text = text.replace("ctton", "cotton")
    text = text.replace("ryaon", "rayon")
    return text


with open("./data/inputs.txt") as f:
    inputs = f.readlines()

with open("./data/outputs.json") as f:
    outputs = json.load(f)

assert len(inputs) == len(outputs)

questions = []

for x, y in zip(inputs, outputs):
    q = []
    x = replace(x)
    x = re.sub("[^a-zA-Z0-9 \n\.]", " ", x)
    x = re.sub("\s\s+", " ", x)
    x = f"[NONE] {x.lower()}"

    # Sort the components according to the order in which they appear in the sentence.
    components = {component: x.find(component) for component in y.keys()}
    components = dict(sorted(components.items(), key=lambda item: item[1]))

    # Ask for components
    for index, component in enumerate(components.keys()):
        # Sometimes there isn't any component.
        # We will ask the model to return [NONE].
        if not component:
            component = "[NONE]"

        if x.find(component) > -1:
            q.append(
                {
                    "answers": {"answer_start": [x.find(component)], "text": [component]},
                    "context": x,
                    "question": f"What is the {index + 1} component ?",
                }
            )
        # We want to avoid missing a component and asking the model to retrieve the n+1
        # component while requesting the nth component.
        else:
            break

    questions += q

    # Ask for materials
    for component, materials in y.items():
        if not component:
            component = "[NONE]"

        for index, material in enumerate(materials):
            proportion = int(material["proportion"])
            material = material["material"]
            found = False

            # We match patterns like "20 cotton" and "20 cotton".
            # We keep both the proportion and the material in the answer to avoid
            # calling the pattern twice. We will extract the proportion and material
            # from the answer later.
            for pattern in [f"{material} {proportion}", f"{proportion} {material}"]:
                if x.find(pattern) > -1:
                    questions.append(
                        {
                            "answers": {"answer_start": [x.find(pattern)], "text": [pattern]},
                            "context": x,
                            "question": f"What is the {index + 1} material of the component {component} ?",
                        }
                    )

test = questions[:500]
train = questions[500:]

with open("data/train.json", "w") as f:
    for row in train:
        json.dump(row, f)
        f.write("\n")

with open("data/test.json", "w") as f:
    for row in test:
        json.dump(row, f)
        f.write("\n")

```
</details>

## Question-answering fine-tuning 

Here is the boring part; I copied and pasted the [Hugging Face code](https://huggingface.co/docs/transformers/tasks/question_answering) dedicated to training question-answering models. It seems that learning Pytorch is no longer helpful for NLP. 

I chose to fine-tune the extractive question answering model [deepset/tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2) because it is light and powerful. Note that the model is specialized in English. Training the model for ten epochs and 2139 questions takes 10 minutes on a GPU.

Here is the code dedicated to the specialization of the model on our dataset:

<details>
<summary>Click to see the code.</summary>

```python
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained("deepset/tinyroberta-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2")
dataset = load_dataset("json", data_files={"train": "data/train.json", "test": "data/test.json"})


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

trainer.train()
```

</details>

## Inference using the extractive question-answering model

I uploaded the model to the Hugging Face hub after training it to make it easier to reuse. Its name is [raphaelsty/carbonblog](https://huggingface.co/raphaelsty/carbonblog).

Let's look at how our model generalizes with our template questions on some examples of the test set:

```python
import re

from transformers import pipeline

qa = pipeline(
    "question-answering", model="raphaelsty/carbonblog", tokenizer="raphaelsty/carbonblog"
)

def clean(document):
    """Pre-process the document."""
    document = re.sub("[^a-zA-Z0-9 \n\.]", " ", document)
    document = re.sub("\s\s+", " ", document)
    # [NONE] allows the model to handle missing components.
    document = f"[NONE] {document.lower()}"
    return document
```

Let's start with pre-processing the documents; we remove the special characters. 

```python
document = clean("body: 83% nylon 17% spandex; lace: 86% nylon 14% spandex")
```

We can then interrogate our model on the first mentioned component.

```python
qa({"question": "What is the 1 component ?", "context": document})
```

```json
{"score": 1.0, "start": 7, "end": 11, "answer": "body"}
```

The second component:

```python
qa({"question": "What is the 2 component ?", "context": document})
```

```json
{"score": 0.9999976754188538, "start": 32, "end": 36, "answer": "lace"}
```

The material (and its proportion) most present in the body component:

```python
qa({"question": "What is the 1 material of the component body ?", "context": document})
```

```json
{"score": 0.9999743103981018, "start": 12, "end": 20, "answer": "83 nylon"}
```

The second most present material (and its proportion) in the lace component:

```python
qa({"question": "What is the 2 material of the component lace ?", "context": document})
```

```json
{"score": 0.9999875426292419, "start": 46, "end": 56, "answer": "14 spandex"}
```

## Automatic extraction

We have a model capable of locating components, materials, and proportions based on questions. Here the idea is simple; we will automate the processing of documents.

We need to identify the number of components and the number of materials associated with each component.

In the document below, two times percentages sum to 100, so there are two components and two materials for each. 

```
body: 83% nylon 17% spandex; lace: 86% nylon 14% spandex
```

Following this logic, we can automate the identification of the number of components and the number of materials associated with each of these components. We can then generate question templates to structure the data.

The code below automates the generation of questions for a given document:

<details>
<summary>Click to see the code</summary>

```python
import collections

def n_components_materials(tokenizer, document):
    """Extract the number of components and materials per component."""
    component, material, percentage, n = 0, 0, 0, {}
    for token in tokenizer(document):
        token = token.text
        # Check for percentages.
        if token.isnumeric():
            material += 1
            percentage += float(token)
        # Number of component is equal to the total sum of percentage / 100
        if percentage >= 100:
            component += 1
            n[component] = material
            percentage, material = 0, 0
    return n


def extract(tokenizer, qa, document):
    """Extract fields from the document."""
    answers = collections.defaultdict(list)
    document = clean(document)
    for component, materials in n_components_materials(
        tokenizer=tokenizer, document=document
    ).items():
        # Ask about the component
        component = qa({"question": f"What is the {component} component ?", "context": document})
        component = component["answer"]

        for material in range(1, materials + 1):
            # Ask about the material and the proportion
            answer = qa(
                {
                    "question": f"What is the {material} material of the component {component} ?",
                    "context": document,
                }
            )
            material, score = answer["answer"], answer["score"]

            # Extract proportion
            proportion = re.findall(r"\d+", material)[0]
            material = material.replace(proportion, "").strip()
            answers[component].append(
                {"material": material, "proportion": float(proportion), "score": round(score, 4)}
            )
    return dict(answers)
```
</details>


```python
import spacy

tokenizer = spacy.load("en_core_web_sm")

document = "body: 83% nylon 17% spandex; lace: 86% nylon 14% spandex"

extract(tokenizer=tokenizer, qa=qa, document=document)
```

```json
{
    "body": [
        {"material": "nylon", "proportion": 83.0, "score": 1.0},
        {"material": "spandex", "proportion": 17.0, "score": 1.0},
    ],
    "lace": [
        {"material": "nylon", "proportion": 86.0, "score": 1.0},
        {"material": "spandex", "proportion": 14.0, "score": 1.0},
    ],
}
```


## Entity linking

We are almost there. The last step is to perform entity linking on the components and materials. To do this, we can use the library Cherche [shameless self-promotion](https://github.com/raphaelsty/cherche). Finally, we use a simple tf-idf for components and materials to try to correct any typos in the documents.

We must define all the components and materials to proceed to the entity linking phase.


```python
components = [
    # The [NONE] component is dedicated to missing components.
    {"component": "", "label": "[NONE]"},
    {"component": "front_panel", "label": "front panel"},
    {"component": "shell", "label": "shell"},
    {"component": "crochet", "label": "crochet"},
    {"component": "g-string", "label": "g-string"},
    {"component": "panty", "label": "panty"},
    {"component": "bottom", "label": "bottom"},
    {"component": "striped_mesh", "label": "striped mesh"},
    {"component": "marl_fabric", "label": "marl fabric"},
    {"component": "tank", "label": "tank"},
    {"component": "centre_front_and_wings", "label": "centre front and wings"},
    {"component": "ank", "label": "ank"},
    {"component": "string", "label": "string"},
    {"component": "top_body", "label": "top body"},
    {"component": "body_panty", "label": "body panty"},
    {"component": "mesh", "label": "mesh"},
    {"component": "pant", "label": "pant"},
    {"component": "aol", "label": "aol"},
    {"component": "cup_shell", "label": "cup shell"},
    {"component": "ruffle", "label": "ruffle"},
    {"component": "elastic", "label": "elastic"},
    {"component": "lace", "label": "lace"},
    {"component": "fabric", "label": "fabric"},
    {"component": "liner", "label": "liner"},
    {"component": "body", "label": "body"},
    {"component": "top", "label": "top"},
    {"component": "forro", "label": "forro"},
    {"component": "lining", "label": "lining"},
    {"component": "rib", "label": "rib"},
    {"component": "cup_lining", "label": "cup lining"},
    {"component": "back_panel", "label": "back panel"},
    {"component": "trim_lace", "label": "trim lace"},
    {"component": "micro", "label": "micro"},
    {"component": "cami", "label": "cami"},
    {"component": "gusset", "label": "gusset"},
    {"component": "edge_lace", "label": "edge lace"},
    {"component": "short", "label": "short"},
    {"component": "knitted_top", "label": "knitted top"},
    {"component": "pants", "label": "pants"},
]

materials = [
    {"material": "cotton", "label": "cotton"},
    {"material": "organic cotton", "label": "organic cotton"},
    {"material": "modal", "label": "modal"},
    {"material": "nylon", "label": "nylon"},
    {"material": "elastane", "label": "elastane"},
    {"material": "eco vero rayon", "label": "eco vero rayon"},
    {"material": "regular polyamide", "label": "regular polyamide"},
    {"material": "polyester", "label": "polyester"},
    {"material": "recycled polyamide", "label": "recycled polyamide"},
    {"material": "rayon", "label": "rayon"},
    {"material": "recycled nylon", "label": "recycled nylon"},
    {"material": "recycled polyester", "label": "recycled polyester"},
    {"material": "acrylique", "label": "acrylique"},
    {"material": "spandex", "label": "spandex"},
    {"material": "polyamide", "label": "polyamide"},
    {"material": "viscose", "label": "viscose"},
    {"material": "lycra", "label": "lycra"},
    {"material": "cotton woven top", "label": "cotton woven top"},
    {"material": "polyester knitted", "label": "polyester knitted"},
    {"material": "ecovero viscose", "label": "ecovero viscose"},
    {"material": "bamboo", "label": "bamboo"},
    {"material": "metallic yarn", "label": "metallic yarn"},
    {"material": "recycled cotton", "label": "recycled cotton"},
]
```

We can then instantiate our entity linking models:

```python
from cherche import retrieve
from sklearn.feature_extraction.text import TfidfVectorizer

retriever_material = retrieve.TfIdf(
    key="material",
    on=["label"],
    documents=materials,
    tfidf=TfidfVectorizer(lowercase=True, ngram_range=(3, 10), analyzer="char_wb"),
)

retriever_component = retrieve.TfIdf(
    key="component",
    on=["label"],
    documents=components,
    tfidf=TfidfVectorizer(lowercase=True, ngram_range=(3, 10), analyzer="char_wb"),
)
```

The tf-idf retriever associated with ngrams can correct some of the typos:

```python
# recycled polyester
retriever_material("plyestr recy")
```

```json
[
    {"material": "recycled polyester", "similarity": 0.39},
    {"material": "polyester", "similarity": 0.28016},
    {"material": "recycled cotton", "similarity": 0.21709},
    {"material": "recycled nylon", "similarity": 0.21686},
    {"material": "polyester knitted", "similarity": 0.19697},
    {"material": "recycled polyamide", "similarity": 0.17656},
    {"material": "regular polyamide", "similarity": 0.02398},
]
```

## Overall pipeline

Here we are. Below is the complete pipeline, consisting of an extractive question-answering model and an entity disambiguation pipeline.

All the code dedicated to inference:

<details>
<summary>Click to see the code</summary>

```python
import collections
import re

import spacy
from cherche import retrieve
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

qa = pipeline('question-answering', model="raphaelsty/carbonblog", tokenizer="raphaelsty/carbonblog")

tokenizer = spacy.load("en_core_web_sm")

def clean(document):
    """Pre-process the document."""
    document = re.sub("[^a-zA-Z0-9 \n\.]", " ", document)
    document = re.sub("\s\s+" , " ", document)
    # [NONE] allows the model to handle missing components.
    document = f"[NONE] {document.lower()}"
    return document

def n_components_materials(tokenizer, document):
    """Extract the number of components and materials per component."""
    component, material, percentage, n = 0, 0, 0, {}
    for token in tokenizer(document):
        token = token.text
        # Check for percentages.
        if token.isnumeric():
            material += 1
            percentage += float(token)
        # Number of component is equal to the total sum of percentage / 100
        if percentage >= 100:
            component += 1
            n[component] = material
            percentage, material = 0, 0
    return n


def extract(tokenizer, qa, document):
    """Extract fields from the document."""
    answers = collections.defaultdict(list)
    document = clean(document)
    for component, materials in n_components_materials(tokenizer=tokenizer, document=document).items():
        # Ask about the component
        component = qa({'question': f"What is the {component} component ?", 'context': document})
        component = component["answer"]

        for material in range(1, materials + 1):
            # Ask about the material and the proportion
            answer = qa({'question': f"What is the {material} material of the component {component} ?", 'context': document})
            material, score = answer["answer"], answer["score"]

            # Extract proportion
            proportion = re.findall(r'\d+', material)[0]
            material = material.replace(proportion, "").strip()
            answers[component].append({"material": material, "proportion": float(proportion), "score": round(score, 4)})
    return dict(answers)


components = [
    # Component [NONE] -> ""
    {'component': '', 'label': '[NONE]'},
    {'component': 'front_panel', 'label': 'front panel'},
    {'component': 'shell', 'label': 'shell'},
    {'component': 'crochet', 'label': 'crochet'},
    {'component': 'g-string', 'label': 'g-string'},
    {'component': 'panty', 'label': 'panty'},
    {'component': 'bottom', 'label': 'bottom'},
    {'component': 'striped_mesh', 'label': 'striped mesh'},
    {'component': 'marl_fabric', 'label': 'marl fabric'},
    {'component': 'tank', 'label': 'tank'},
    {'component': 'centre_front_and_wings', 'label': 'centre front and wings'},
    {'component': 'ank', 'label': 'ank'},
    {'component': 'string', 'label': 'string'},
    {'component': 'top_body', 'label': 'top body'},
    {'component': 'body_panty', 'label': 'body panty'},
    {'component': 'mesh', 'label': 'mesh'},
    {'component': 'pant', 'label': 'pant'},
    {'component': 'aol', 'label': 'aol'},
    {'component': 'cup_shell', 'label': 'cup shell'},
    {'component': 'ruffle', 'label': 'ruffle'},
    {'component': 'elastic', 'label': 'elastic'},
    {'component': 'lace', 'label': 'lace'},
    {'component': 'fabric', 'label': 'fabric'},
    {'component': 'liner', 'label': 'liner'},
    {'component': 'body', 'label': 'body'},
    {'component': 'top', 'label': 'top'},
    {'component': 'forro', 'label': 'forro'},
    {'component': 'lining', 'label': 'lining'},
    {'component': 'rib', 'label': 'rib'},
    {'component': 'cup_lining', 'label': 'cup lining'},
    {'component': 'back_panel', 'label': 'back panel'},
    {'component': 'trim_lace', 'label': 'trim lace'},
    {'component': 'micro', 'label': 'micro'},
    {'component': 'cami', 'label': 'cami'},
    {'component': 'gusset', 'label': 'gusset'},
    {'component': 'edge_lace', 'label': 'edge lace'},
    {'component': 'short', 'label': 'short'},
    {'component': 'knitted_top', 'label': 'knitted top'},
    {'component': 'pants', 'label': 'pants'}
]


materials = [
    {'material': 'cotton', 'label': 'cotton'},
    {'material': 'organic cotton', 'label': 'organic cotton'},
    {'material': 'modal', 'label': 'modal'},
    {'material': 'nylon', 'label': 'nylon'},
    {'material': 'elastane', 'label': 'elastane'},
    {'material': 'eco vero rayon', 'label': 'eco vero rayon'},
    {'material': 'regular polyamide', 'label': 'regular polyamide'},
    {'material': 'polyester', 'label': 'polyester'},
    {'material': 'recycled polyamide', 'label': 'recycled polyamide'},
    {'material': 'rayon', 'label': 'rayon'},
    {'material': 'recycled nylon', 'label': 'recycled nylon'},
    {'material': 'recycled polyester', 'label': 'recycled polyester'},
    {'material': 'acrylique', 'label': 'acrylique'},
    {'material': 'spandex', 'label': 'spandex'},
    {'material': 'polyamide', 'label': 'polyamide'},
    {'material': 'viscose', 'label': 'viscose'},
    {'material': 'lycra', 'label': 'lycra'},
    {'material': 'cotton woven top', 'label': 'cotton woven top'},
    {'material': 'polyester knitted', 'label': 'polyester knitted'},
    {'material': 'ecovero viscose', 'label': 'ecovero viscose'},
    {'material': 'bamboo', 'label': 'bamboo'},
    {'material': 'metallic yarn', 'label': 'metallic yarn'},
    {'material': 'recycled cotton', 'label': 'recycled cotton'}
]


retriever_material = retrieve.TfIdf(
        key = "material",
        on = ["label"],
        documents = materials,
        tfidf = TfidfVectorizer(lowercase=True, ngram_range=(3, 10), analyzer="char_wb")
)

retriever_component = retrieve.TfIdf(
    key = "component",
    on = ["label"],
    documents = components,
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(3, 10), analyzer="char_wb")
)

def parse(tokenizer, qa, document, retriever_material, retriever_component):
    """Ask questions and retrieves components and materials using Cherche."""
    try:
        items = extract(tokenizer=tokenizer, qa=qa, document=document)
    except:
        raise ValueError(f"Error parsing document:\n\t{document}")

    components_materials = collections.defaultdict(list)

    for component, materials in items.items():

        # Retrieve the right component.
        component_found = retriever_component(component)
        if component_found:
            component_found = component_found[0]["component"]
        else:
            raise ValueError(f"Unable to retrieve component document:\n\t{document}")

        for material in materials:

            # Retrieve the right material.
            material_found = retriever_material(material["material"])
            if material_found:
                material_found = material_found[0]["material"]
                components_materials[component_found].append({"material": material_found, "proportion": material["proportion"], "score": material["score"]})
            else:
                raise ValueError(f"Unable to retrieve component document:\n\t{document}")

    return dict(components_materials)
```
</details>

Example of application of the pipeline on a set of various descriptions:

```python
documents = [
    "top body: 100% polyester lace: 88% nylon 12% spandex, string: 88% nylon 12% spandex",
    "92% polyester, 8% spandex",
    "95% rayon 5% spandex" "lace 87% nylon 13% spandex; mesh: 95% nylon 5% spandex",
    "body & panty: 85% nylon 15% spandex",
    "86%polyamide,14%elastane",
]

for document in documents:
    print(
        parse(
            tokenizer=tokenizer,
            qa=qa,
            document=document,
            retriever_material=retriever_material,
            retriever_component=retriever_component,
        )
    )
```

Pipeline predictions:

```json
{
    "body": [{"material": "polyester", "proportion": 100.0, "score": 1.0}],
    "lace": [
        {"material": "nylon", "proportion": 88.0, "score": 1.0},
        {"material": "spandex", "proportion": 12.0, "score": 1.0},
    ],
    "string": [
        {"material": "nylon", "proportion": 88.0, "score": 1.0},
        {"material": "spandex", "proportion": 12.0, "score": 0.8396},
    ],
}
{
    "": [
        {"material": "polyester", "proportion": 92.0, "score": 1.0},
        {"material": "spandex", "proportion": 8.0, "score": 1.0},
    ]
}
{
    "": [
        {"material": "rayon", "proportion": 95.0, "score": 1.0},
        {"material": "spandex", "proportion": 5.0, "score": 1.0},
    ],
    "lace": [
        {"material": "nylon", "proportion": 87.0, "score": 1.0},
        {"material": "spandex", "proportion": 13.0, "score": 1.0},
    ],
    "mesh": [
        {"material": "nylon", "proportion": 95.0, "score": 0.6435},
        {"material": "spandex", "proportion": 5.0, "score": 0.9969},
    ],
}
{
    "body_panty": [
        {"material": "nylon", "proportion": 85.0, "score": 1.0},
        {"material": "spandex", "proportion": 15.0, "score": 1.0},
    ]
}
{
    "": [
        {"material": "polyamide", "proportion": 86.0, "score": 1.0},
        {"material": "elastane", "proportion": 14.0, "score": 1.0},
    ]
}
```

## Evaluation

An evaluation of the results of the pipeline on 500 documents (test set) shows that the pipeline finds 96% of the fields on average. I leave the detailed analysis of the model's errors to another time.

|    Field   | Precision |
|:----------:|:---------:|
|  component |   96.61   |
|  material  |   96.19   |
| proportion |   96.32   |

Below is the code associated with the pipeline evaluation:

<details>
<summary>Click to see the code</summary>

```python
import collections
import json

import tqdm
from river import stats

with open("./data/inputs.txt") as f:
    inputs = f.readlines()

# Remove \n
inputs = [x.replace("\n", "") for x in inputs]

with open("./data/outputs.json") as f:
    outputs = json.load(f)

assert len(inputs) == len(outputs)

missed = []

accuracy_components = stats.Mean()
accuracy_materials = stats.Mean()
accuracy_proportions = stats.Mean()

error_components = collections.defaultdict(int)
error_materials = collections.defaultdict(int)
error_proportions = collections.defaultdict(int)


for x, y in tqdm.tqdm(zip(inputs[:500], outputs[:500]), position=0):

    try:
        y_pred = parse(
            tokenizer=tokenizer,
            qa=qa,
            document=x,
            retriever_material=retriever_material,
            retriever_component=retriever_component,
        )
    except:
        missed.append(x)

    for component, materials in y.items():
        if component in y_pred:
            accuracy_components.update(1)
            for idx, material in enumerate(materials):
                if material["material"] == y_pred[component][idx]["material"]:
                    accuracy_materials.update(1)
                else:
                    accuracy_materials.update(0)
                    error_materials[material["material"]] += 1

                if material["proportion"] == y_pred[component][idx]["proportion"]:
                    accuracy_proportions.update(1)
                else:
                    accuracy_proportions.update(0)
                    error_proportions[material["proportion"]] += 1

        else:
            accuracy_components.update(0)
            error_components[component] += 1
            # We are wrong without the right field.
            for material in materials:
                accuracy_materials.update(0)
                accuracy_proportions.update(0)
                
print(
    f"Precision: components {accuracy_components.get():2f}, materials {accuracy_materials.get():2f}, proportion {accuracy_proportions.get():2f}"
)
```
</details>

## There is room for improvement

There is plenty of room for improvement. This contribution represents a baseline for further improvements. It can also help facilitate the labeling of new data. The model would benefit from various data-augmentation strategies from knowledge bases like Wordnet. 

Unfortunately, I have not incorporated a feedback loop into this model. For example, we could study the errors and confidence scores of the model in cross-validation to create such a procedure.

I appreciated the problem that Max posed. In the context of my thesis in NLP at Renault and Université Paul Sabatier, I regularly think about methods for generalization on datasets with a particular vocabulary (different from Wikipedia) and with a reduced number of training data. I think my answer shows that pre-trained extractive question-answering models associated can identify simple patterns with few examples on real-world datasets. 

