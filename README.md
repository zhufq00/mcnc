Experiment code for: "A Generative Approach for Script Event Prediction via Contrastive Fine-tuning", AAAI 23.

Including main experiments and ablation experiments and the datasets we used. All experiments can be run directly through the corresponding configuration file when appropriate paths are set in the configuration file.

## 1.  Datasets
The public dataset: can be found here:  [GitHub - eecrazy/ConstructingNEEG_IJCAI_2018: The data and code for our IJCAI-ECAI 2018 paper: Constructing Narrative Event Evolutionary Graph for Script Event Prediction](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018)

```plain
cd mcnc
mkdir data
mkdir data/negg_data
# negg_data_dir represents the address of the downloaded dataset
mv negg_data_dir/corpus_index_dev.txt ./data/negg_data/dev
mv negg_data_dir/corpus_index_test.txt ./data/negg_data/test
mv negg_data_dir/corpus_index_train0.txt ./data/negg_data/train
```
We also uploaded this dataset along with the code.
The original dataset: due to dataset upload size limitations, we will publish this dataset after the paper is published.

## 2.  Experiments
### 2.1 main experiments:

Bart_base_with_public_dataset($Bart_{base}$):  

```plain
# 1. stage1: event-centric pretraining
python main.py --config_file ./configs/main_experiments/bart_base_event-centric_pretraining.yaml
# 2. stage2: task-specific contrastive fine-tuning
# check ./log/date/bart_base_event-centric_pretraining.log and copy the last
# "best_checkpoint_path" to the "checkpoint_path" item of ./configs/main_experiments/bart_base_contrastive_fine-tuning.yaml
python main.py --config_file ./configs/main_experiments/bart_base_contrastive_fine-tuning.yaml
# check performance in ./log/date/bart_base_event-centric_pretraining.log
```

A similar process for repetition $Bart_{large}$and $Bart_{large}+NYT$model.

### 2.2 ablation experiments:

An example for ablation method *replace_with_a_linear_classifier.*

```plain
# copy the above "best_checkpoint_path" path of Bart_base_with_public_dataset
# to the "checkpoint_path" item of ./configs/ablation_experiments/replace_with_a_linear_classifier.yaml
python main.py --config_file ./configs/ablation_experiments/replace_with_a_linear_classifier.yaml
# check performance in ./log/date/replace_with_a_linear_classifier.log
```

### 2.3 checkpoint:

The final checkpoint for ours($Bart_{base}$) on the public dataset, if you need more checkpoint, log, or dataset please contact me, I'll try to find it out, but I can't promise since I've cleared hard drive space.

https://drive.google.com/file/d/1nWnDc-E6mo42G4wycbkk2UpK6JBuEWaX/view?usp=sharing

