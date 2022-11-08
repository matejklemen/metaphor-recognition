# metaphor-detection

## Data format

An assumption is made on the data format. It should be a tab-separated file with the following columns:  
- `document_name`: ID of the document that the current sentence belongs to (set dummy IDs if there are no documents). 
This is used to extract previous sentences to provide the model additional context.
- `idx`: position of the current sentence in the current document.
- `sentence_words`: words in the sentence (list of strings)
- `met_type`: annotated metaphors in the document (optional)

See [cjvt/komet](https://huggingface.co/datasets/cjvt/komet), [cjvt/gkomet](https://huggingface.co/datasets/cjvt/gkomet), 
and [matejklemen/vuamc](https://huggingface.co/datasets/matejklemen/vuamc) for examples of datasets formatted in this way.
Below is a snippet of how the data should look like:
```
document_name	idx	idx_paragraph	idx_sentence	sentence_words	met_type	met_frame	met_type_mapped
komet1484.div.xml	60	31	0	['»', 'Jaz', 'grem', 'tudi', 'sam', 'domov', '!', '«', 'je', 'Maticu', 'zasijal', 'obraz', '.']	[{'type': 'MRWi', 'word_indices': [10]}]	[{'type': 'cause_visual_change', 'word_indices': [10]}]
komet1484.div.xml	61	31	1	['»', 'Imaš', 'kaj', 'drobiža', '?', '«']	[]	[]
```

## Training the models
Token-level:
```shell
$ python3 metaphor_detection_token.py \
--mode="train" \
--experiment_dir="komet-fold1-xlmrbase-2e-5-binary3-history0-optthresh" \
--train_path=data/komet-5fold/fold1/train.tsv  \
--dev_path=data/komet-5fold/fold1/dev.tsv \
--test_path=data/komet-5fold/fold1/test.tsv \
--history_prev_sents=0 \
--pretrained_name_or_path="xlm-roberta-base" \
--learning_rate=2e-5 \
--batch_size=120 \
--num_epochs=10 \
--validate_every_n_examples=3000 \
--early_stopping_rounds=100 \
--validation_metric="f1_score_binary" \
--random_seed=17 \
--type_scheme="binary" \
--mrwi \
--mrwd \
--mrwimp \
--wandb_project_name="metaphor-komet-token-span-optimization" \
--optimize_bin_threshold
```

Sentence-level:  
```shell
$ python3 metaphor_detection_sentence.py \
--mode="train" \
--experiment_dir="komet-sent-fold0-sloberta-2e-5-binary3-history0-optthresh" \
--train_path=data/komet-sent-5fold/fold0/train.tsv \
--dev_path=data/komet-sent-5fold/fold0/dev.tsv \
--test_path=data/komet-sent-5fold/fold0/test.tsv \
--history_prev_sents=0 \
--pretrained_name_or_path="EMBEDDIA/sloberta" \
--learning_rate=2e-5 \
--batch_size=200 \
--num_epochs=10 \
--validate_every_n_examples=3000 \
--early_stopping_rounds=100 \
--validation_metric="f1_score_binary" \
--random_seed=17 \
--mrwi \
--mrwd \
--mrwimp \
--wandb_project_name="metaphor-komet-sentence" \
--optimize_bin_threshold
```