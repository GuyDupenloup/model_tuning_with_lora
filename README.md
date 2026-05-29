# GPT-2 Model Tuning


## 1. Introduction

In a previous project, I created a GPT-2 model from scratch using the original Transformer paper and the GPT/GPT-2 papers from OpenAI.

See Github repo: [GPT-2 Model From Research Papers](https://github.com/GuyDupenloup/gpt2_model_from_research_papers)

In this project, I experimented with tuning a GPT-2 model for the following applications:

- Question answering using the *SQuAD* dataset
- Text simplification using the *wikilarge* dataset (the cleaned version of it)
- News classification using the *ag_news* dataset

My goals were as follows:

1. Create a GPT-2 model with built-in LoRA adapters, along with the full pipeline required to train and evaluate models, and run inference using batches of prompts that use different adapters.

2. Get hands-on experience with fine-tuning a model to perform multiple tasks, comparing sequential fine-tuning of the same model and multiple LoRA adapters.

3. Run prompts through the model with LoRA adapters and analyze results.

The code I wrote to conduct experiments was designed to be straightforward to run, with hardcoded directory structure, file names, and flow handling. However, it can be easily extended or modified to support additional tasks and datasets, and many components can be reused in different environments.

## 2. Project setup

### 2.1 Source code

The source code for this project is in the **./src** directory and is organized as shown below.

```
    src
     |     
     ├── models
     |     ├── gpt2_model.py               # GPT-2 base model with built-in LoRA layers
     |     └── gpt2_language_model.py      # GPT-2 language model (base model with LM head)
     |
     ├── utils
     |     ├── dataset_utils.py            # Export datasets to TFRecords, create data loaders
     |     ├── model_utils.py              # Create and load models, load OpenAI weights, get model summaries
     |     └── gen_text.py                 # Generate texts from prompts
     |
     └── scripts
           ├── squad_dataset.py            # Preprocess SQuAD dataset and export to TFRecords
           ├── wikilarge_dataset.py        # Prepare`wikilarge dataset and export to TFRecords
           ├── ag_news_dataset.py          # Preprocess ag_news dataset and export to TFRecords
           ├── openai_weights.py           # Get OpenAI GPT-2 model weights and save them to numpy arrays
           ├── train.py                    # Train models independently or sequentially
           ├── train_lora.py               # Train models using LoRA adapters
           └── test_prompts.py             # Load a trained model with LoRA adapters and run prompts through it
```

### 2.2 Python packages

I used the *transformers* package from Hugging Face to get OpenAI GPT-2 weights. Using this package requires TensorFlow version 2.14.1 or older. The packages I used are listed in file **requirements_1.txt**.

Once extracted from the Hugging Face model, weights are saved to numpy arrays. Then, you can switch to more recent versions of TensorFlow to train, evaluate and test the models. The packages I used for that are listed in file **requirements_2.txt**.

### 2.3 Python search path

To run the scripts, you need to add the **src** directory path to the PYTHONPATH environment variable that sets the search path for Python, as shown below.

```bash
# Linux
export PYTHONPATH="/mypath/src:$PYTHONPATH"

# Windows cmd
set PYTHONPATH=%PYTHONPATH%;C:\mypath\src
```

## 3. Executing the project

### 3.1 Directory structure and files

When you prepare the datasets and train the models, directories and files will get created under a **project root** directory as shown in the diagram below. In the repository, the project root is **src/project**, but you can use your own name and location.

The diagram only shows directories and files that are created when using the 124M model size. If you use a different size, for example 355M, a directory called **gpt2_355M** will be created under the project root with the same structure and files as shown in the diagram.

Datasets are preprocessed and exported to TFRecords.

Each saved model consists of two files:
- a JSON configuration file used to recreate the model
- a weights file to load into the model

```
<project root>
     |     
     ├── datasets
     |     |
     |     ├── squad                           # Created by script `squad_dataset.py`
     |     |     |
     |     |     ├── metadata.json
     |     |     └── train.tfrecord, val.tfrecord, test.tfrecord
     |     |
     |     ├── wikilarge                       # Created by script `wikilarge_dataset.py`
     |     |     |
     |     |     ├── metadata.json
     |     |     └── train.tfrecord, val.tfrecord, test.tfrecord
     |     |
     |     └── ag_news                         # Created by script `ag_news_dataset.py`
     |           |
     |           ├── metadata.json
     |           └── train.tfrecord, val.tfrecord, test.tfrecord
     |
     └── gpt2_124M
           |
           ├── pretrained_weights
           |        |
           |        └── openai_weights_gpt2_124M.npz
           |
           ├── trained_models
           |        | 
           |        ├── squad_baseline.json, squad_baseline.weights.h5   # Created by script `train.py`
           |        ├── wikilarge_baseline.json, wikilarge_baseline.weights.h5
           |        ├── ag_news_baseline.json, ag_news_baseline.weights.h5
           |        |
           |        └── lora_adapters.json, lora_adapters.weights.h5    # Created by script `train_lora.py`
           |
           └── tests
                 |
                 ├── example_prompts.json
                 └── model_responses.txt          # Created by script `test_prompts.py`
```

### 3.2 Running the project scripts

To execute the project, run the scripts that are in the **src/scripts** directory as shown below.

```bash

# Step 1: create the project root directory
export PROJECT=/home/user/myproject
mkdir $PROJECT

# Step 2: go to the directory that contains the scripts
cd src/scripts

# Step 3: prepare the datasets
python squad_dataset.py --project_root $PROJECT
python wikilarge_dataset.py --project_root $PROJECT
python ag_news_dataset.py --project_root $PROJECT

# Step 4: export OpenAI weights to numpy arrays
python openai_weights.py --model_size 124M --project_root $PROJECT

# Step 5: run baseline trainings (independent training of three different models)
python train.py --model_size 124M --project_root $PROJECT --run baseline

# Step 6: run sequential trainings of the same model
python train.py --model_size 124M --project_root $PROJECT --run sequential

# Step 7: train LoRA adapters
python train_lora.py --model_size 124M --project_root $PROJECT

# Step 8: test example prompts with LoRA adapters
python test_lora.py --project_root $PROJECT
```

If you are not interested in the sequential training experiments, you can skip steps 5 and 6.


## 4. GPT-2 model enhancements

I made the following enhancements to the GPT-2 model I created in my previous project:

- Built-in Low-Rank Adaptation (LoRA) layers inside the multi-head attention blocks and feed-forward network

- Mechanism to load OpenAI pretrained weights into the model when LoRA adapters are present

- Loss mask to exclude the prompt from loss calculation, keeping only the output part

- Language modelling head that includes loss and metrics calculation, and LoRA adapters management

For LoRA adapters, I used the architecture described in the original paper published by Edward J. Hu, Yelong Shen, et al in 2021:

[LoRA: Low-Rank Adaptation of Large Language Models.](https://arxiv.org/abs/2106.09685).


## 5. Model tasks, datasets and prompts

### 5.1 Loss and attention masks

All the prompts start with a "### Task: " header that indicates the task the model has to perform. Tasks include "answer question" when training and evaluating with the SQuAD dataset, "simplify text" with wikilarge, and "classify news" with ag_news. 

I used **tiktoken** for tokenization which does not have a dedicated \<EOS\> token, so I used the pad token 50256 to mark the end of the model answers.

The loss mask hides the entire prompt for loss calculation, only keeping the model answer. This way, the model is only rewarded on its answer, not on the prompt. The attention mask hides all the pad tokens that come after the ending pad token used as the \<EOS\> token. Note that it is crucial that the loss mask and attention mask both keep the ending pad token as it informs the model of where to end its answer.

### 5.2 SQuAD dataset

Examples from the SQuAD dataset are formatted as shown below. The prompt ends after "### Answer: " and is followed by the model answer. The pad token used as < EOS > token is shown as "<|endoftext|>".

```
### Task: answer question

### Context: The performance of "Summertime" by Barrino, later known simply as "Fantasia", at Top 8 was widely praised, and Simon Cowell considered it as his favorite Idol moment in the nine seasons he was on the show. Fantasia and Diana DeGarmo were the last two finalists, and Fantasia was crowned as the winner. Fantasia released as her coronation single "I Believe", a song co-written by season one finalist Tamyra Gray, and DeGarmo released "Dreams". Fantasia went on to gain some successes as a recording artist, while Hudson, who placed seventh, became the only Idol contestant so far to win both an Academy Award and a Grammy.

### Question: What was Fantasia's coronation song?

### Answer: I Believe<|endoftext|>
```

### 5.3 wikilarge dataset

Examples from the wikilarge dataset are formatted as shown below. The prompt ends after "### Simplified: " and is followed by the model answer.

```
### Task: simplify text

### Text: He is considered by some historians to have had a crucial influence on the transition of New South Wales from a penal colony to a free settlement and therefore to have played a major role in the shaping of Australian society in the early nineteenth century .

### Simplified: Historians say he led the change of New South Wales from a penal colony to a free settlement .<|endoftext|>
```


### 5.4 ag_news dataset

Examples from the ag_news dataset are formatted as shown below. The prompt ends after "### Label: " and is followed the model answer, which is either "Business', "Sports", "Sci/Tech", or "World".

```
### Task: classify news

### News: Office Depot cuts profit forecast Office Depot Inc. warned Tuesday of weaker-than-expected profits for the rest of the year because of disruptions from the string of hurricanes in Florida and poor sales in the rest of North America and in Europe.

### Label: Business<|endoftext|>
```

## 6. Model training

### 6.1 Metrics

To evaluate the performance of the models, I used *exact-match accuracy* for the SQuAD and ag_news datasets, and perplexity for the wikilarge dataset.


### 6.2 Baseline training

Before training a model with LoRA adapters, I needed a baseline to serve as a performance reference.

Therefore, I created three GPT-2 models that I trained independently on respectively SQuAD, wikilarge, and ag_news. I did not freeze any layers for these trainings, so all the pretrained weights were trainable. It may be possible to get better results by freezing some layers.

The table below summarizes the results I obtained with these baseline trainings.


|   dataset           |  Test set before training  |  Training set    |  Validation set  |  Test set   |
|---------------------|----------------------------|------------------|------------------|-------------|
|   SQuAD             |         0.0                |       46.9       |      45.8        |   45.8      | 
|   wikilarge         |          6.80              |     3.56         |      3.60        |   3.34      | 
|   ag_news           |         0.0                |       94.4       |      94.2        |   92.1      |


### 6.3. Sequential training

Having established performance references with the baseline trainings, the next step is to investigate how the model performs when it is trained sequentially on the three datasets.

I ran a number of the following experiments:

1. A model is created and fine-tuned on a given dataset (the baseline).
2. The model is trained on a different dataset.
3. The model is re-evaluated on the first dataset.

If the model performs well on the test sets of the two datasets, it is capable of handling the two corresponding tasks.

The results I obtained with these experiments are summarized in the tables below.


|   Experiment #1                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Create model and train on SQuAD              |             |                  |    45.8      |
|   2. Train model on wikilarge                     |  3.58       |      3.62        |    3.40      |
|   3. Re-evaluate model on SQuAD                   |             |                  |    6.80      |


|   Experiment #2                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Create model and train on wikilarge          |             |                  |     3.34     |
|   2. Train model on SQuAD                         |   50.5      |     46.2         |     46.2     |
|   3. Re-evaluate model on wikilarge               |             |                  |     7.86     |


|   Experiment #3                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Create model and train on SQuAD              |             |                  |    45.8      |
|   2. Train model on ag_news                       |     95.0    |     94.2         |    94.0      |
|   3. Re-evaluate model on SQuAD                   |             |                  |    0.0       |


|   Experiment #4                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Create model and train on ag_news            |             |                  |    92.1      |
|   2. Train model on SQuAD                         |     47.17   |    44.2          |    44.2      |
|   3. Re-evaluate model on ag_news                 |             |                  |    52.6      |


|   Experiment #5                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Create model and train on wikilarge          |             |                  |    3.34      |
|   2. Train model on ag_news                       |   95.0      |       93.8       |    93.8      |
|   3. Re-evaluate model on wikilarge               |             |                  |    49.14     |


|   Experiment #6                                   |  Train set  |  Validation set  |  Test set    |
|---------------------------------------------------|-------------|------------------|--------------|
|   1. Create model and train on ag_news            |             |                  |    92.1      |
|   2. Train model on wikilarge                     |    3.62     |     3.7120       |    3.41      |
|   3. Re-evaluate model on ag_news                 |             |                  |    4.17      |


All these experiments show that attempting to fine-tune the same model sequentially on several datasets is not a viable solution.

For example in experiment #3, the model trained on SQuAD starting from pretrained weights (the baseline) has a test set accuracy of 45.8%. Then, when the model is trained on the ag_news dataset, it reaches the ag_news baseline accuracy at ~94.0%. But when the model is re-evaluated on SQuAD, accuracy has dropped from 45.8% to 0.0%. In other words, from the SQuAD dataset perspective, the model went back to where it was before training. We have here a case of so-called *catastrophic forgetting*.

### 6.4 LoRA adapters training

Next, I trained three LoRA adapters, each one being responsible for a given task.

The results I obtained are summarized in the table below.


|   LoRA adapter      |  Trainable parameters  |  Train set  |  Validation set  |  Test set  |
|---------------------|------------------------|-------------|------------------|------------|
|   SQuAD             |  2.21M (rank=16)       |    39.7     |   44.2           |    44.2    |
|   wikilarge         |  1.1M (rank=8)         |    4.15     |   3.95           |    3.35    |
|   ag_news           |  1.1M (rank=8)         |    92.9     |    93.0          |    93.0    |

For SQuAD, the adapter reaches 44.2% versus 45.8% for the baseline (I could get to the baseline with larger adapters, but with diminishing results). For wikilarge, the adapter reaches the same perplexity as the baseline at ~3.3. For ag_news, the adapter achieves 93.0% versus 92.4% for the baseline.

Note that all adapters are more or less under-fitted, which is probably a consequence of the small number of trainable parameters of each adapter.

Using these three adapters, the model performs at about the same levels as the baselines. They only add 4.4M parameters to the initial model that has 124M parameters, representing only a 3.5% increase. This is quite remarkable and demonstrates the effectiveness of the LoRA approach.

Additionally, the LoRA adapters don't alter the model weights. If all adapters are disabled, the model reverts to the original pretrained GPT-2 behavior.


## 7. Testing and analyzing prompt responses

### 7.1 Testing prompts

I tested the model with LoRA adapters using 50 examples for each adapter. I extracted them from the test sets of the datasets, which the model did not see during training. They are in the JSON file **\<project root\>/gtpt2_124M/prompt_tests/example_prompts.json**. 

The script **src/scripts/test_prompts.py** activates the adapters, runs the example prompts through the model, and collects the model responses. Then, it writes to file **\<project root\>/gtpt2_124M/prompt_tests/model_responses.txt** each example comprising of:

- A unique example ID
- The prompt followed by the model response
- The reference answer from the dataset

For the question answering and news classification tasks, I used greedy sampling. For text simplification, I used temperature=0.8 and top-k sampling with k=20.

### 7.2 Question answering (SQuAD test set)

**Accuracy metric**:

One obvious observation is that the exact-match accuracy metric is often too crude to reflect the actual performance of the model, and the 45.8% accuracy I obtained under-estimates it. The model does not get any credit for answers that are correct but formulated differently than the references, answers that are more or less verbose than the reference, and answers that are correct but incomplete.

Prompt IDs 10, 19, 21, 36, 40, 44, and 47 are examples of this issue.

**Model strengths**:

- **Factual recall on clean questions**: IDs 4, 5, 22, 24, 25, 26, 32, 34, 35, 39 are all correct and well-formed. The model handles straightforward who/what/when questions reliably.

- **Appropriate answer brevity**: The model generally extracts compact spans and avoids copying entire sentences. For example in ID 19, it gives a straight-to-the-point answer while the reference is too verbose. However, it sometimes cuts off its answer too soon, like in ID 24 where it answers "16th" instead of "16th century", or in ID 8 where it answers "go home" instead of "go home and change".

- **Semantic understanding**: ID 29 ("separation" for "fragmentation") shows that the model grasps meaning even when it doesn't match the reference exactly.

- **Robust to varied writing styles and text structures**: The model is able to locate relevant spans in contexts written in a scientific, journalistic, or historical register. 

**Model weaknesses**:

- **Confusion between entities of the same type**: When a sentence contains multiple entities of the same type (e.g., two different years, two different radio stations, or two different numbers), the model often picks the wrong one from the immediate vicinity. This happens for example in ID 2 where the model picks the "KOA" radio station instead of "KRFX", and in ID 27 where it chooses "Disney–ABC International Television" instead of "Disney–ABC Domestic Television".

- **Weak numerical reasoning**: In ID 18, the context includes "about twice as much (14.6 mg·L−1) dissolves at 0 °C than at 20 °C.". When asked "How much more oxygen dissolves at 0 degrees C than at 20 degrees C?", the model fails on basic logic and answers "14.6 mg·L−1" instead of "twice".

### 7.3 Text simplification (wikilarge test set)

**Model strengths:**

- **Minor rephrasing and trimming**: The model handles simple simplifications well, removing text between parentheses, introductory phrases, or redundant clauses without distorting meaning. IDs 54, 55, 58, 73, 84, 98, and 99 are clean examples where the output is fluent and faithful to the source.

- **Vocabulary substitution**: The model occasionally replaces words with simpler synonyms appropriately. In ID 79, "interred" becomes "buried", and in ID 91, "substantial" becomes "large".

- **Sentence splitting**: In ID 81, the model correctly splits and reorganizes a complex sentence into a main clause and a subordinate clause ("When he is going to rehearsal..."), which functions as a legitimate simplification strategy.

**Hallucinations**:

The model frequently "hallucinates", generating content that is entirely absent from the source text or fabricating new facts.

- ID 50: claims the character "lives with many people" with no basis in the source.

- ID 56: invents a completely fictional claim that the topic is "the oldest museum in Scotland, and the largest in England."

- ID 57: replaces the named entity "Saturn" with a completely unmentioned person, "Ventura".

- ID 61: adds that giardiasis is caused specifically "in small children" with no factual backing in the prompt.

- ID 93: completely fabricates a sentence about bird population metrics ("The number of birds in the population has doubled since 2000.") out of a source text about marine life (pipefish and seahorses).

**Meaning-altering omissions**:

The model sometimes drops information that changes the meaning or completely strips out the core subject rather than simplifying it.

- ID 63: drops the phrase "subject of numerous reports", oversimplifying the text to the point of a meaning shift ("The event was called ethics in scholarship.").

- ID 71: drops "off the northern coast of Haiti", producing a much less geographically informative sentence.

- ID 76: cuts the second half of the sentence entirely, missing the key requirement that a user must set a nickname to connect to IRC.

- ID 89: strips away almost all context regarding the survival of a brand across digital media, reducing it to a vague statement about what it was named after.

- ID 95: drops the entire subject of the sentence ("real estate, businesses and other assets in the underground economies of the Third World"), leaving a dangling pronoun ("They cannot be used...") with no referent.

**Loss of referential clarity**:

- In ID 67, changing "Landis' father" to just "Landis" ruins the referential clarity, making it sound like the son is supporting himself.

- In ID 90, changing "Schuschnigg immediately responded publicly that..." to "This led to..." deletes the actor entirely, making it unclear who or what caused the reports to be false.

- In ID 97, the model shifts from a general statement about five Dravidian languages to a highly specific sentence focusing only on "The Tamil language," losing the broader scope of the original message.

**Numerical errors**: In ID 82, the model inexplicably changes "Seventy-five defencemen" to "Thirty-four defencemen", introducing a blatant factual error.

**No simplification**: In IDs 59, 62, 64, 65, and 72, the model outputs the source sentence verbatim, performing no simplification at all.


### 7.4 News classification (ag_news test set)

The model demonstrates high accuracy, giving 46 correct answers to the 50 news to classify:

- Sports: 14/14
- World: 12/12
- Business: 12/13
- Sci/Tech: 8/11

In IDs 102, 105 and 112, the model chooses Sci/Tech instead of Business. These examples all involve technology (e.g. network equipment, Cisco Systems, software service, computerized service, Computer Associates), which probably is the reason why the model got confused.

In ID 115, the news is about researchers discovering that some diseases are influenced by gender. The model answers Sci/Tech while the reference is World. Arguably, the model is correct and the reference wrong.

## 8. Conclusion

This project gave me hands-on experience with the full fine-tuning pipeline for a GPT-2 model, from dataset preparation to training, evaluation, inference, and prompt analysis.

The sequential training experiments clearly illustrated the catastrophic forgetting problem: fine-tuning a single model on multiple tasks sequentially destroys previously acquired capabilities. LoRA adapters provide an efficient solution. Using three adapters totaling 4.4M parameters, which represents only a 3.5% of the base model parameters, the model achieves near-baseline performance on all three tasks simultaneously, without touching the original weights during adapter training.

The prompt analysis revealed that raw accuracy metrics can be misleading. The SQuAD model's 45.8% exact-match accuracy underestimates its actual usefulness, while the wikilarge model's good perplexity score does not reflect its strong tendency to hallucinate. This exercise showed me how essential qualitative analysis is alongside quantitative metrics.
