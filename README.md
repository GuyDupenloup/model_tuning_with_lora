# GPT-2 Model Tuning


## 1. Introduction

In a previous project, I created a GPT-2 model from scratch using the original Transformer paper and the GPT/GPT-2 papers from OpenAI.

See Github repo: [GPT-2 Model From Research Papers](https://github.com/GuyDupenloup/gpt2_model_from_research_papers)

In this project, I experimented with tuning a GPT-2 model for the following applications:

- Question answering using the *SQuAD* dataset
- Text simplification using the *wikilarge* dataset (the cleaned version of it)
- News classification using the *ag_news* dataset

My goal was twofold:

1. Create a GPT-2 model with built-in LoRA adapters, along with the full pipeline required to train and evaluate models, and manage adapters.

2. Get hands-on experience with tuning a model to perform multiple tasks, comparing sequential fine-tuning of the same model and LoRA adapters.

The code I wrote to conduct experiments was designed to be straightforward to run, with hardcoded file and flow handling. Although the objective was not to create a configurable environment, it can be easily extended or modified to support additional tasks and datasets. Many components of the code are modular and reusable across projects, including dataset preparation, data loaders, training routines, and text generation utilities.


## 2. Project setup

### 2.1 Source code

The source code for this project is in the *./src* directory and is organized as shown below.

```
    src
     |     
     ├── models
     |     ├── gpt2_model.py               # GPT-2 base model with built-in LoRA layers
     |     └── gpt2_language_model.py      # GPT-2 language model (base model with LM head)
     |
     ├── utils
     |     ├── dataset_utils.py            # Format prompts, export datasets to TFRecords, create data loaders
     |     ├── model_utils.py              # Create models, load OpenAI weights, get model summaries
     |     └── gen_text.py                 # Generate texts from prompts
     |
     └── scripts
           |
           ├── squad_dataset.py            # Preprocess `SQuAD` dataset and export to TFRecords
           ├── wikilarge_dataset.py        # Prepare `wikilarge` dataset and export to TFRecords
           ├── ag_news_dataset.py          # Preprocess `ag_news` dataset and export to TFRecords
           ├── openai_weights.py           # Get OpenAI GPT-2 model weights and save them to numpy arrays
           ├── train.py                    # Train models independently or sequentially
           ├── train_lora.py               # Train a model using LoRA adapters
           └── test.py                     # Test prompts using LoRA models
```

### 2.2 Python packages

I used the *transformers* package from Hugging Face to get OpenAI GPT-2 weights. Using this package requires version 2.14.1 of TensorFlow, or older. The packages I used are listed in file *requirements_1.txt*.

Once extracted from the Hugging Face model, OpenAI weights are saved to numpy arrays. Then, you can switch to more recent versions of TensorFlow to train, evaluate and test the models. The packages for that I used are listed in file *requirements_2.txt*.

### 2.3 Python search path

To run the scripts, you need to add the *src* directory path to the PYTHONPATH environment variable that sets the search path for Python, as shown below:

```bash
# Linux
export PYTHONPATH="/mypath/src:$PYTHONPATH"

# Windows cmd
set PYTHONPATH=%PYTHONPATH%;C:\mypath\src
```

## 3. Executing the project

### 3.1 Directory structure and files

When you prepare the datasets and train the models, scripts will create the directory structure and files shown below. The diagram only shows directories and files for the 124M model.

```
< project root >
     |     
     ├── datasets
     |     |
     |     ├── squad                          # Created by script `squad_dataset.py`
     |     |     |
     |     |     ├── metadata.json
     |     |     ├── train.tfrecord
     |     |     ├── val.tfrecord
     |     |     └── test.tfrecord
     |     |
     |     ├── wikilarge                       # Created by script `wikilarge_dataset.py`
     |     |     |
     |     |     ├── metadata.json
     |     |     ├── train.tfrecord
     |     |     ├── val.tfrecord
     |     |     └── test.tfrecord
     |     |
     |     └── ag_news                         # Created by script `ag_news_dataset.py`
     |     |     |
     |     |     ├── metadata.json
     |     |     ├── train.tfrecord
     |     |     ├── val.tfrecord
     |     |     └── test.tfrecord
     |
     ├── openai_weights                        # Created by script `openai_weights.py`
     |      |
     |      └── openai_weights_gpt2_124M.npz
     |
     └── trained_models_124M
            |
            ├── baseline                       # Created by script `train.py`
            |     ├── squad_baseline.json
            |     ├── squad_baseline.weights.h5
            |     ├── wikilarge_baseline.json
            |     ├── wikilarge_baseline.weights.h5
            |     ├── ag_news_baseline.json
            |     └── ag_news_baseline.weights.h5
            |
            ├── sequential                      # Created by script `train.py`
            |     ├── sequential_1.json
            |     ├── sequential_1.weights.h5
            |     ├── sequential_2.json
            |     └── sequential_2.weights.h5
            | 
            └── lora                            # Created by script `train_lora.py`
                  ├── lora.json
                  └── lora.weights.h5
```

### 3.2 Running the project scripts

To execute the project, run the scripts that are in the *./src/scripts* directory as shown below.

```bash
# Step 1: create the project root directory
export PROJECT=/home/user/myproject
mkdir $PROJECT

# Step 2: export OpenAI weights to numpy arrays
python openai_weights.py --model_size 124M --weights_filepath $PROJECT/openai_weights/openai_weights_gpt2_124M.npz

# Step 3: prepare the datasets
python squad_dataset.py --project_root $PROJECT
python wikilarge_dataset.py --project_root $PROJECT
python ag_news_dataset.py --project_root $PROJECT

# Step 4: run baseline trainings (independent training of 3 different models)
python train.py --model_size 124M --project_root $PROJECT --run baseline

# Step 5: run sequential trainings of the same model
python train.py --model_size 124M --project_root $PROJECT --run sequential

# Step 6: train LoRA adapters
python train_lora.py --model_size 124M --project_root $PROJECT

# Step 7: test example prompts with LoRA model
python test_prompt.py --project_root $PROJECT --model_dir <directory where the model was saved> --prompt <your prompt here>
```

If you are not interested in the sequential training experiments, you can skip steps 4 and 5.


## 4. GPT-2 model enhancements

I made the following enhancements to my original GPT-2 model:

- Built-in Low-Rank Adaptation (LoRA) layers inside the multi-head attention blocks and feed-forward network

- Mechanism to make OpenAI weights loadable into the model in the presence of LoRA adapters

- Loss mask to only take into account the model output part of the prompt when computing the loss

- Language modelling head that includes loss and metrics calculation, and LoRA adapters management

For LoRA adapters, I used the architecture described in the original paper published by Edward J. Hu, Yelong Shen, et al in 2021:

[LoRA: Low-Rank Adaptation of Large Language Models.](https://arxiv.org/abs/2106.09685).

I also added LoRA layers to the feed-forward network as they can contribute significantly to the model performance. Any number of adapters can be inserted in the model. Each of them includes a dropout layer.


## 5. Model tasks, datasets and prompts

The prompts are formatted to handle the different tasks the model will have to perform. The first part "### Task: " specifies the task to perform.

I used tiktoken for tokenization. As it has no dedicated < EOS > character, I used the pad token 50256 to mark the end of the model response. It appears as *<|endoftext|>* in the examples below.


### 5.1 SQuAD dataset

Examples from the SQaD dataset are formatted as shown below. The prompt ends after "### Answer: " and is followed by the model answer.

```
### Task: answer question

### Context: The performance of "Summertime" by Barrino, later known simply as "Fantasia", at Top 8 was widely praised, and Simon Cowell considered it as his favorite Idol moment in the nine seasons he was on the show. Fantasia and Diana DeGarmo were the last two finalists, and Fantasia was crowned as the winner. Fantasia released as her coronation single "I Believe", a song co-written by season one finalist Tamyra Gray, and DeGarmo released "Dreams". Fantasia went on to gain some successes as a recording artist, while Hudson, who placed seventh, became the only Idol contestant so far to win both an Academy Award and a Grammy.

### Question: What was Fantasia's coronation song?

### Answer: I Believe<|endoftext|>
```

During training:

- The loss mask excludes the prompt from loss calculation and only keeps the model answer part, which is "I Believe<|endoftext|>" in the example above. This forces the model to focus on the answer.

- The attention mask uncovers the prompt and the model answer, and masks the padding tokens that comes afterwards to avoid that the model attends to them.

- Both the loss mask and attention mask uncover the padding token that marks the end of the answer, as the model must be made aware that this is where the answer ends.


### 5.2 Wikilarge dataset

Examples from the *wikilarge* dataset are formatted as shown below. The prompt ends after "### Simplified: " and is followed by the model answer.

```
### Task: simplify text

### Text: He is considered by some historians to have had a crucial influence on the transition of New South Wales from a penal colony to a free settlement and therefore to have played a major role in the shaping of Australian society in the early nineteenth century .

### Simplified: Historians say he led the change of New South Wales from a penal colony to a free settlement .<|endoftext|>
```

The loss mask only keeps what comes after "### Simplified: ", i.e. the model answer.


### 5.3 ag_news dataset

Examples from the *ag_news* dataset are formatted as shown below. The prompt ends after "### Label: " and is followed the model answer, which is either "Business', "Sports", "Sci/Tech", or "World".

```
### Task: classify news

### News: Office Depot cuts profit forecast Office Depot Inc. warned Tuesday of weaker-than-expected profits for the rest of the year because of disruptions from the string of hurricanes in Florida and poor sales in the rest of North America and in Europe.

### Label: Business<|endoftext|>
```

## 6. Model training

### 6.1 Training setup

I only ran the smallest GPT-2 model that has 124M parameters.

To evaluate the performance of the models, I used *label-level exact match accuracy* for the *SQuAD* and *ag_news* datasets, and perplexity for the *wikilarge* dataset.

In the result tables below, accuracy values was multiplied by 100.


### 6.2 Baseline training

|   dataset           |  Test set before training  |  Training set    |  Validation set  |  Test set   |
|---------------------|----------------------------|------------------|------------------|-------------|
|   SQuAD             |         0.0                |       46.9       |      45.8        |   45.8      | 
|   wikilarge         |          6.80              |     3.56         |      3.60        |   3.34      | 
|   ag_news           |         0.0                |       94.4       |      94.2        |   92.1      |   <= good



### 6.3. Sequential training

Having established a baseline, we know what the model can do best on each dataset individually. The next step is to determine whether the model can perform at the same level when it is trained sequentially on the three datasets, so that it is perform the three tasks we task we want: answering a question, simplifying a text, and classifying news.

To determine that, I did a number of experiments. First, a model trained on a given dataset is loaded, i.e. what I called the "baseline model". Then, the model is trained on another dataset. Finally, the model is re-evaluated on the dataset used for the baseline. If the model performs well on the test sets of the two datasets, then it is capable of handling both tasks.


|   Experiment #1                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Load SQuAD baseline model                    |             |                  |    45.8      |
|   2. Train model on wikilarge                     |  3.58       |      3.62        |    3.40      |
|   3. Re-evaluate model on SQuAD                   |             |                  |    6.80      |


|   Experiment #2                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Load wikilarge baseline model                |             |                  |     3.40     |
|   2. Train model on SQuAD                         |   50.5      |     46.2         |     46.2     |
|   3. Re-evaluate model on wikilarge               |             |                  |     7.86     |


|   Experiment #3                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Load SQuAD baseline model                    |             |                  |    45.8      |
|   2. Train model on ag_news                       |     95.0    |     94.2         |    94.0      |
|   3. Re-evaluate model on SQuAD                   |             |                  |    0.0       |


|   Experiment #4                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Load ag_news baseline model                  |             |                  |    92.4      |
|   2. Train model on SQuAD                         |     47.17   |    44.2          |    44.2      |
|   3. Re-evaluate model on ag_news                 |             |                  |    52.6      |


|   Experiment #5                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Load wikilarge baseline model                |             |                  |    3.40      |
|   2. Train model on ag_news                       |   95.0      |       93.8       |    93.8      |
|   3. Re-evaluate model on wikilarge               |             |                  |    49.14     |


|   Experiment #6                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Load ag_news baseline model                  |             |                  |    92.4      |
|   2. Train model on wikilarge                     |    3.62     |     3.7120       |    3.41      |
|   3. Re-evaluate model on ag_news                 |             |                  |    4.17      |


All these experiments show that training the same model sequentially on the three datasets is not a viable solution.

For example in experiment #3, the model trained on SQuAD starting from OpenAI weights (the baseline) has a test set accuracy of 45.0%. Then, when the model is trained on the ag_news dataset, it reaches the ag_news baseline accuracy at ~94.0%. But when re-evaluating the model on SQuAD, accuracy has dropped from 45.0% to 0%. In other words, from the SQuAD perspective, the model went back to to where it was before training. This is a case of so called "catastrophic forgetting".

### 6.4 LoRA adapters training

Next, i trained three LoRA adapters, each one being responsible for a given task.

The results I obtained are summarized in the table below.


|   LoRA adapter      |  Trainable parameters  |  Train set  |  Validation set  |  Test set  |
|---------------------|------------------------|-------------|------------------|------------|
|   SQuAD             |  2.21M (rank=16)       |    39.6     |   43.6           |    43.6    |
|   wikilarge         |  1.1M (rank=8)         |    4.15     |   3.9507         |    3.33    |
|   ag_news           |  1.1M (rank=8)         |    92.9     |    94.1          |    93.8    |

For SQuAD, the adapter reaches 43.6% versus 45.0% for the baseline. I could come back to about the same accuracy as the baseline with larger adapters, but returns were diminishing.

For wikilarge, the adapter reaches the same perplexity as the baseline.

For ag_news, the adapter reaches 93.8% versus 92.4% for the baseline, so slightly better.

Note that all adapters are more or less under-fitted, which is probably a consequence of the small number of trainable parameters of each adapter.

Using these three adapters, the model performs at about the same levels as the baselines. They only add 4.4M parameters to the initial model that has 124M parameters, representing only 3.5%. This is quite remarkable and demonstrates the effectiveness of the LoRA approach.

Additionally, the LoRA adapters don't alter OpenAI's parameters. If all adapters are disconnected from the model, it performs like it does when OpenAI parameters are not tuned for a given application.


## 7. Testing and analyzing prompt responses

### 7.1 Testing prompts
For each task, I tested the model with LoRA adapters using 50 examples randomly extracted from the test sets.

The examples are in file example_prompts.json.

File lora_responses.txt was created but script test_prompts.py that runs the prompts through the model.They include:
- A unique example ID
- The prompt with the model response
- The annotation from the dataset

There are a number of observations that can be made looking at prompts with their model answers. 

### 7.2 Question answering (SQuAD test set)

#### Accuracy metric

A first, obvious observation is that the exact-match accuracy metric is often too crude to reflect the actual performance of the model, and the 45% accuracy I obtained under-estimates the model.

The model does not get any credit for answers that are correct but formulated differently than the annotations, answers that are more or less verbose than the annotation, and answers that are correct but incomplete.

Prompt IDs 10, 19, 8 and 21 are examples of this issue.

#### Model strengths

The model demonstrates the following strengths:

- Factual recall on clean questions. IDs 4, 5, 22, 24, 25, 26, 32, 34, 35, 39 are all correct and well-formed. The model handles straightforward who/what/when questions reliably.

- Appropriate answer brevity. The model generally extracts compact spans and avoids copying entire sentences. For example in ID 19, it gives a straight-to-the-point answer while the annotation is too verbose. However, it sometimes cuts off its answer too soon, like in ID 24 where it answers "16th" instead of "16th century", or in ID 8 where it answers "go home" instead of "go home and change".

- Semantic understanding. ID 29 ("separation" for "fragmentation") and ID 33 show that the model grasps meaning even when it doesn't match the annotation exactly.

- Robust to varied writing styles and text structures. Whether the context is written in a scientific, journalistic, or historical register, it can still locate relevant spans. 

#### Confusion between entities of the same type

When a sentence contains multiple entities of the same type (e.g., two different years, two different radio stations, or two different numbers), the model often picks the wrong one from the immediate vicinity.

For example in ID 2, the model picks the "KOA" radio station instead of "KRFX". The same type of confusion occurs in ID 27 where the model chooses "Disney–ABC International Television" instead of "Disney–ABC Domestic Television".

#### Numerical values reasoning

In ID 18, the context includes "about twice as much (14.6 mg·L−1) dissolves at 0 °C than at 20 °C.". When asked "How much more oxygen dissolves at 0 degrees C than at 20 degrees C?", the model answers "14.6 mg·L−1" instead of "twice".

The model fails on this question that only requires basic counting and logic. This weakness may be a consequence of the limited model capacity (only 124M parameters).


### 7.3 Text simplification (Wikilarge test set)

#### Model strengths

The model demonstrates the following strengths:

- Minor rephrasing and trimming. The model handles simple simplifications well, removing text between parenthesis, introductory phrases, or redundant clauses without distorting meaning. IDs 54, 62, 63, 72, 84, 97 are clean examples where the output is fluent and faithful to the source.

- Vocabulary substitution. The model occasionally replaces words with simpler synonyms appropriately. In ID 70, "tends to be unaware" becomes "is unaware", and in ID 91, "regions" becomes "areas" and "substantial" becomes "significant".

- Sentence splitting. In ID 60, the model correctly splits a complex sentence into two simpler ones, which is a legitimate simplification strategy.

#### Hallucinations

The model also frequently "hallucinates", generating content that is entirely absent from the source text:

- ID 52: adds "in a car accident" with no basis in the source.
- ID 55: invents "1982 American Athletic Conference championship".
- ID 57: replaces all named entities with "Tiger Woods" three times.
- ID 59: replaces "Rancho Palos Verdes" with "Rambo Palos Verdes" and fabricates a description.
- ID 83: invents a claim about the "highest-ever percentage of votes".
- ID 92: adds "in Israel" with no basis.
- ID 98: replaces "Brighton" with "Blaze".

This critical weakness affects a large portion of outputs.

#### Meaning-altering omissions

The model sometimes drops information that changes the meaning rather than simplifying it:

- ID 71: drops "organized into a tropical depression off the northern coast of Haiti", producing a much less informative sentence.
- ID 77: drops the key detail that it was Tazz, not Venis, who hit Rikishi with the camera, altering who did what.
- ID 96: drops the 1994 tour entirely.

#### Numerical errors

In ID 93, the model changes "forty-nine" to "thirty-nine" with no justification, introducing a factual error.

#### Copying without simplifying

In IDs 87 and 95, the model outputs the source sentence verbatim, performing no simplification at all.
Loss of referential clarity. In ID 79, "He is buried there" loses the specific location entirely, making the sentence less informative than the original.

Overall, the model is far less reliable than the SQuAD model. While it handles minor edits competently, its hallucination rate is high enough to make it untrustworthy for any real simplification task.

### 7.4 News classification (ag_news test set)

The model demonstrates the following strengths:

- High overall accuracy. The model gets the vast majority of labels correct. Only a handful of errors are visible across 50 examples, which suggests the model has learned the four-class classification task (World, Business, Sports, Sci/Tech) quite well.

- Handles ambiguous cases reasonably. Some articles sit at the boundary between two categories, and the model's errors are understandable rather than random:

      - ID 102: the model answers "Business" for a news article about Cisco acquiring a company for $200 million. This is a defensible answer even though the annotation is "Sci/Tech", since the article is primarily about a financial transaction.
      - ID 105: the model answers "Sci/Tech" for a story about a phone spoofing service, while the annotation is "Business". Again, a reasonable confusion.
      - ID 112: the model answers "Sci/Tech" for a story about a Computer Associates legal case, while the annotation is "Business". The company name likely triggered a technology association.

- Robust across writing styles. The model handles headlines, wire service dispatches, and longer articles equally well, suggesting it learned category-level signals rather than surface formatting cues.

However, the model also exhibits the following weaknesses:

- Business/Sci/Tech boundary confusion. The clearest failure pattern is confusing these two categories, which is unsurprising since technology companies frequently appear in business news. IDs 102, 105, and 112 all fall into this pattern.

- One surprising error. ID 115 ("Men, Women More Different Than Thought") is labeled "Sci/Tech" by the model but annotated as "World". This is arguably the annotation's fault rather than the model's — a research finding about gender differences fits "Sci/Tech" more naturally than "World". This echoes the exact-match accuracy problem you identified for the SQuAD model.

### 8. Conclusion about the abilities of the model

- SQuAD (Extractive QA)

The model shows genuine capability on this task. It reliably extracts relevant spans for straightforward factual questions, and its errors are often minor — wrong verbosity level, slightly truncated answers, or confusion between entities of the same type. The 45% exact-match accuracy meaningfully underestimates the model's true performance due to the crudeness of the metric. The model is arguably usable for simple QA applications where answers are unambiguous and entities are not repeated in the context, though it should not be trusted for questions requiring numerical reasoning.

- WikiLarge (Text Simplification)
This is where the model fails most severely. While it handles minor edits competently — dropping redundant clauses, light vocabulary substitution — its hallucination rate is far too high for any practical use. Fabricating facts (IDs 52, 55, 57, 92, 98), altering named entities, and introducing false information make the model untrustworthy for this task. Text simplification may be an inherently harder task for a model of this size, as it requires both understanding the source sentence and generating a faithful rewrite, rather than simply extracting or classifying. GPT-2 small is not usable for this application.

- AG News (News Classification)
This is the task where the model performs best and most consistently. The four-class classification problem appears well within the model's capacity, with errors concentrated in the naturally ambiguous Business/Sci/Tech boundary. The model is arguably production-ready for this task, with the caveat that some annotation noise makes it hard to distinguish true model errors from labeling disagreements.

- Overall conclusion
The results suggest that GPT-2 small's usefulness scales inversely with task complexity. Classification, which only requires the model to map an input to a fixed label, is well within its capacity. Span extraction is manageable but imperfect. Open-ended generation, as required by text simplification, exposes the model's core limitation: it cannot reliably stay faithful to a source text while rewriting it, producing hallucinations that make it unsuitable for real-world use. For any application where factual faithfulness matters, a larger model or a different architecture would be necessary.

