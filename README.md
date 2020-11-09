# CxGBERT: BERT meets Construction Grammar

This repository contains data and code associated with the publication  *CxGBERT: BERT meets Construction Grammar"* (COLING 2020). 

In the paper we show that **BERT has access to a information that linguists typically call constructional**. 

For sentences tagged with constructional information, see [this folder](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/tree/master/C2xG#tagged-sentences).

## Table of Contents
 - [What is Construction Grammar](#What-is-Construction-Grammar)
 - [Why Construction Grammar and BERT?](#Why-Construction-Grammar-and-BERT?)
 - [Preprocessing](#Preprocessing)
 - [Creating the Pre-Training and Probing Data](#Creating-the-Pre-Training-and-Probing-Data)
 - [Constructional Test Data](#Constructional-Test-Data)
 - [Evaluating BERT's ability to detect Constructions](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/blob/master/README.md#evaluating-berts-ability-to-detect-constructions)
 - [Pre-training Hyperparameters](#Pre-training-Hyperparameters)
 - [Pre-Trained Models](#Pre-Trained-Models)

## What is Construction Grammar

Construction Grammars are a set of linguistic theories that take constructions as their primary object of study. Constructions are either patterns that are frequently occurring or their meaning is not predictable from the sum of their parts (e.g. Idioms). This study focuses more on constructions that consist of frequently occuring patterns. 

These patterns can be  Low level and simple: Noun +s (Plural Construction) 
Another low level construction is *The Xer the Yer*. Some sentences that are instances of this construction include *a) The more you think about it, the less you understand, b) The Higher You Fly, the Further You Fall*

An example of a higher level pattern is  *Personal Pronoun + didn’t + V + how* and sentences that are instances of this construction include: *a) She didn’t understand how I could do so poorly, and b) One day she picked up a book and as she opened it, a white child took it away from her, saying she didn’t know how to read.*

There are also *schematic constructions* such as the *Caused-Motion Construction*. Some instances: *a) Norman kicked the ball into the room. b) Mary hit the ball out of the park. c) She sneezed the foam off the cappuccino.*

## Why Construction Grammar and BERT?

Consider the final example in the previous section *She sneezed the foam off the cappuccino.* It has been shown that humans can understand sentences with such novel uses of words (or novel words) using the construction the sentences are instances of. 

Given that BERT has access to PoS information, parse trees, and mBERT can even reproduce labels for syntactic dependencies that largely agree with universal grammar, we ask: **How much constructional information does BERT have access to?**


## Preprocessing

These experiments require a large corpus to be tagged with constructional information. We do this using version of C2xG and details including pre-tagged data is available in the [C2xG Folder](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/tree/master/C2xG). 

## Creating the Pre-Training and Probing Data

To create data for pre-training a clone of BERT Base, CxGBERT and BERT Random, please run scripts available in the folder [CreatePreTrainAndEvaluationData](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/tree/master/CreatePreTrainAndEvaluationData). 

 - [CxGCreateData.sh](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/blob/master/CreatePreTrainAndEvaluationData/CxGCreateData.sh) provides the parameters used to run [createCxGData.py](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/blob/master/CreatePreTrainAndEvaluationData/createCxGData.py). 
 - [CxGFeatureSelect.sh](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/blob/master/CreatePreTrainAndEvaluationData/CxGFeatureSelect.sh) can be used to get an estimate of the number of sentences contained in constructions. This should not be required unless to you changing the evaluation metrics.

## Constructional Test Data

The probing data used to test BERT (unadulterated by constructional information) is also generated by the above scripts and is available in [THIS](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/tree/master/CxGTestData) folder. 

**You can use this data to test your models ability to distinguish between sentences that belong to the same construction or not**

The folder contains the subfolders each of which contain train, test and dev files associated with constructions which have that many sentences as instances: 10000-6000000  1000-10000  100-1000  2-10000  2-50  2-6000000  50-100

Each training, test and dev files are TSV files in the same format as the MRPC data: Label, Sent1_id, Sent2_id, Sent1, Sent 2

Here is an example: 

	1  2396.0  2396.1  In 1926 , the tradition of Elephant Walk began when two seniors in the band led a procession of seniors throughout the school grounds visiting all the important places on campus .  In response to protests from seniors , he amended the plan in April 2009 to reduce both the income level at which seniors would have to start paying and the amount which those seniors would have to pay .
	0  543.31  2558.16  The first third of the novel provides a lengthy exploration of the characters ' histories .  "Grahame @-@ Smith met with Keaton in February 2012 , "" We talked for a couple of hours and talked about big picture stuff ."

The first sentence pair belong to the same construction (id 2396) and the second pair do not. 

You can run the following shell script to create the inoculation data described in the paper: 

```bash

inoculate() {

    data_location=$source/$data_dir
    full_out_path=$outloc/$data_dir-$inoc
    mkdir -p $full_out_path

    head -n $inoc $data_location/train.tsv > $full_out_path/train.tsv
    cp $data_location/dev.tsv $full_out_path/dev.tsv
    cp $data_location/test.tsv $full_out_path/test.tsv

}

export source=</path/to/CxGTestData>
export outloc=</paht/to/cxg_inoc_output>
mkdir -p $outloc

export data_dirs='2-50 100-1000 1000-10000 10000-6000000 2-10000 2-6000000 50-100'
for data_dir in $data_dirs
do
    export inocs='100 500 1000 5000'
    for inoc in $inocs
    do
        inoculate
    done
done
```

## Evaluating BERT's ability to detect Constructions

Data generated in the previous section is used to evaluate BERT's ability to detect constructions. Since **the constructional data is in the same format as MRPC, any model that can run on MRPC (part of GLUE dataset) can be run on this data**. The hyperparameters we use are as follows: 

```bash
export output_dir=gs://<paht/to/output/> 
python3 run_classifier.py \ 
	--task_name=MRPC \ # Cause CxG data is in this format
	--do_train=true \ 
	--do_eval=true \ 
	--data_dir=/path/to/CxGTestData \ 
	--vocab_file=gs://<path/to/cased_L-12_H-768_A-12/vocab.txt \ 
	--bert_config_file=gs:<path/to/cased_L-12_H-768_A-12/bert_config.json \ 
	--init_checkpoint=gs://<path/to/cased_L-12_H-768_A-12/bert_model.ckpt \ 
	--save_checkpoints_steps=10000 \ 
	--max_seq_length=128 \ 
	--train_batch_size=32 \ 
	--learning_rate=2e-5 \ 
	--do_lower_case=False \ 
	--num_train_epochs=3 \ 
	--output_dir=$output_dir \ 
	--use_tpu=True \
	--tpu_name=$tpu_name

```

**WARNING:** When evaluating models trained on inoculation data, which contains as few as 100 training examples, it is very important to perform the experiment several times as the results can vary drastically.  

## Pre-training Hyperparameters

We use the original [BERT package](https://github.com/google-research/bert) to pre-train our BERT models. 

Pre-train data is created using: 

	python3 create_pretraining_data.py \
		--do_lower_case=False \
		--input_file=gs://<path/to/txt-file/> \
		--output_file=gs://<path/to/new-train-data.tfrecord> \
		--vocab_file=gs://<path/to/cased_L-12_H-768_A-12/vocab.txt> \
		--random_seed=42 \
		--do_whole_word_mask=True \
		--max_seq_length=128 \
		--max_predictions_per_seq=20 \
		--masked_lm_prob=0.15 \
		--dupe_factor=1

Pre-training is done using 

	python3 run_pretraining.py \
      --input_file=gs://<output/of/previous/command/train-data.tfrecord \
      --output_dir=gs://<path/to/outdir/> \
      --do_train=True \
      --do_eval=True  \
      --bert_config_file=gs://<path/to/cased_L-12_H-768_A-12/bert_config.json> \
      --train_batch_size=512 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --save_checkpoints_steps=20000 \
      --learning_rate=2e-4 \
      --use_tpu=True \
      --num_train_steps=100000 \
      --num_warmup_steps=1000 \
      --tpu_name=<tpu-name>


## Pre-Trained Models

We will be releasing the pre-trained models through HuggingFaces Transformers very soon. 

