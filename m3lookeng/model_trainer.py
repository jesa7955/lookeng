# coding=utf-8
"""BERT finetuning task."""

import collections
import logging
import os
import random

import pandas as pd
import gokart
import luigi
import jaconv
from pyknp import Juman
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from absa_bert_pair import tokenization
from absa_bert_pair.modeling import BertConfig, BertForSequenceClassification
from absa_bert_pair.optimization import BERTAdam

from .data_reader import GenearteSentimentAnalysisData

logger = logging.getLogger('luigi-interface')


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def create_examples(data, split: str):
    return [
        InputExample(
            guid=f'{split}-{index}',
            text_a=tokenization.convert_to_unicode(str(example['sentence1'])),
            text_b=tokenization.convert_to_unicode(str(example['sentence2'])),
            label=tokenization.convert_to_unicode(str(example['label'])))
        for index, example in enumerate(data[split])
    ]


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then
    # each token that's truncated likely contains more information than a
    # longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class FineTuningBertClassifier(gokart.TaskOnKart):
    task_namespace = "m3lookeng"
    # TODO: QA_M, NLI_M, QA_B, NLI_M
    task_name = luigi.Parameter()
    absa_base_path = luigi.Parameter()
    vocab_file = luigi.Parameter()
    bert_config_file = luigi.Parameter()
    init_checkpoint = luigi.Parameter()
    # TODO: whether running eval is needed
    # eval_test = luigi.Parameter()
    # eval_batch_size = luigi.Parameter()
    do_lower_case = luigi.BoolParameter()
    max_seq_length = luigi.IntParameter()
    train_batch_size = luigi.IntParameter()
    learning_rate = luigi.FloatParameter()
    num_train_epochs = luigi.IntParameter()
    warmup_proportion = luigi.FloatParameter()
    # TODO: No CUDA?
    # no_cuda = luigi.BoolParameter()
    accumulate_gradients = luigi.IntParameter()
    gradient_accumulation_steps = luigi.IntParameter()
    # TODO: distributed training?
    # local_rank = luigi.IntParamter()
    seed = luigi.IntParameter()

    def requires(self):
        # TODO
        return GenearteSentimentAnalysisData(
            task_name=self.task_name, absa_base_path=self.absa_base_path)

    def output(self):
        return self.make_model_target(f'absa_bert_pair_{self.task_name}.pkl')

    def run(self):
        data = self.load()
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available()
                                  and not self.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of
            # sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

        logger.info(f"device: {device}, n_gpu: {n_gpu}, distributed training: "
                    f"{bool(self.local_rank) != -1}")

        if self.accumulate_gradients < 1:
            raise ValueError(f"Invalid accumulate_gradients parameter: "
                             f"{self.accumulate_gradients}, should be >= 1")

        self.train_batch_size = int(self.train_batch_size /
                                    self.accumulate_gradients)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        if self.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                f"Cannot use sequence length {self.max_seq_length} because "
                f"the BERT model was only trained up to sequence length "
                f"{bert_config.max_position_embeddings}")

        # prepare dataloaders
        label_lists = {
            "NLI_M": ['neutral', 'positive', 'negative'],
            "QA_M": ['neutral', 'positive', 'negative'],
            "NLI_B": ['0', '1'],
            "QA_B": ['0', '1']
        }

        label_list = label_lists[self.task_name]

        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        # training set
        train_examples = create_examples(data, 'train')

        num_train_steps = int(
            len(train_examples) / self.train_batch_size *
            self.num_train_epochs)

        train_features = convert_examples_to_features(train_examples,
                                                      label_list,
                                                      self.max_seq_length,
                                                      tokenizer)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {self.train_batch_size}")
        logger.info(f"  Num steps = {num_train_steps}")

        all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                       dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features],
                                     dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, all_label_ids)
        if self.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=self.train_batch_size)

        # test set
        if self.eval_test:
            dev_examples = create_examples(data, 'dev')
            dev_features = convert_examples_to_features(
                dev_examples, label_list, self.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in dev_features],
                                         dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in dev_features],
                                          dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in dev_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in dev_features],
                                         dtype=torch.long)

            dev_data = TensorDataset(all_input_ids, all_input_mask,
                                     all_segment_ids, all_label_ids)
            dev_dataloader = DataLoader(dev_data,
                                        batch_size=self.eval_batch_size,
                                        shuffle=False)

        # model and optimizer
        model = BertForSequenceClassification(bert_config, len(label_list))
        if self.init_checkpoint is not None:
            model.bert.load_state_dict(
                torch.load(self.init_checkpoint, map_location='cpu'))
        model.to(device)

        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [{
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate':
            0.01
        }, {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate':
            0.0
        }]

        optimizer = BERTAdam(optimizer_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=num_train_steps)

        # train
        global_step = 0
        epoch = 0
        for _ in trange(int(self.num_train_epochs), desc="Epoch"):
            epoch += 1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            # eval_test
            if self.eval_test:
                model.eval()
                dev_loss, dev_accuracy = 0, 0
                nb_dev_steps, nb_dev_examples = 0, 0
                for input_ids, input_mask, segment_ids, label_ids in dev_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_dev_loss, logits = model(input_ids, segment_ids,
                                                     input_mask, label_ids)

                    logits = F.softmax(logits, dim=-1)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    tmp_dev_accuracy = np.sum(outputs == label_ids)

                    dev_loss += tmp_dev_loss.mean().item()
                    dev_accuracy += tmp_dev_accuracy

                    nb_dev_examples += input_ids.size(0)
                    nb_dev_steps += 1

                dev_loss = dev_loss / nb_dev_steps
                dev_accuracy = dev_accuracy / nb_dev_examples

            result = collections.OrderedDict()
            if self.eval_test:
                result = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': tr_loss / nb_tr_steps,
                    'dev_loss': dev_loss,
                    'dev_accuracy': dev_accuracy
                }
            else:
                result = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': tr_loss / nb_tr_steps
                }

            logger.info("***** Eval results *****")
            logger.info(', '.join(
                ['{key}: value' for key, value in result.items()]))
