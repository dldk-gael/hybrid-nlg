"""
lm_scorer module implements sentence scoring functions that aim to automatically evaluate 
sentences' naturalness by taking advantage of state-of-the-art pretrained language models. 

The module implements two versions : one using GPT2-based models and another one
using BERT-based models.
"""

from abc import ABC, abstractmethod
from typing import *
import math

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np


class SentenceScore(ABC):
    """
    Abstract class for sentence scoring.

    Each class that inherits from SentenceScore will use a particular transformer-based models
    to associate the naturalness of a sentence given some previous context :
    --> naturalness(sentence | context)

    Typically, a scorer based on current state-of-the-art language models works as follows :
    1/ Tokenize the sentence
    2/ Compute the probability of each sentence's tokens for the language model
    3/ Aggregate and normalize tokens' scores

    This class implements various functions common to all lm-based scoring functions
    """

    def __init__(self, model_name: str = "gpt2", batch_size: int = 1, device: str = None, normalization_strategy="LP"):
        """
        - model_name : name of the pretrained model. 
            List of available huggingface's model detailed here : https://huggingface.co/transformers/pretrained_models.html
        - batch_size : maximum number of sentences to input in one single pass into the model 
        - device : where to load the model (GPU, CPU, etc)
        - normalization_strategy : how to aggregate the tokens' scores (LP -> sum of log prob, or MeanLP -> average of log prob)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.normalization_strategy = normalization_strategy
        self.context = None
        self.context_ids: List[int] = []

    def set_context(self, context: str):
        self.context = context
        self.context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"] if self.context else []

    def compute_score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        sentences = [text] if isinstance(text, str) else text

        if self.context_ids != []:
            # Because in BPE, tokenisation is different if there is a space before a word
            sentences = [" " + sentence for sentence in sentences]

        # We can not directly preprend the special tokens because we first have to insert the context
        encodings = self.tokenizer(sentences, add_special_tokens=False)
        scores = self._transformer_log_prob(encodings["input_ids"])

        if self.normalization_strategy == "MeanLP":
            scores = [score / len(encodings.tokens(i)) for (i, score) in enumerate(scores)]

        scores = [math.exp(score) for score in scores]  # All previous computations were performed in log-space
        return scores[0] if isinstance(text, str) else scores

    @abstractmethod
    def _transformer_log_prob(self, sentences_token_ids: List[List[int]]) -> List[float]:
        """
        Given a list of tokenized and encoded sentences
        return the list of log probability of each sentences for the language model
        """
        # Need to be implement for each type of language model (Causal LM, Mask LM, ...)
        ...

    @staticmethod
    def _pad(sequences: List[torch.Tensor], pad_token_id) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rewrite torch.nn.utils.rnn.pad_sequence so that it returns a boolean mask of pad positions.
        The goal is to avoid having to add custom pad tokens to the model and be able to directy pad 
        with any token we want because we still will be able to remove later the score of those tokens.
        """
        max_seq_len = max([s.size(0) for s in sequences])
        out_tensor = sequences[0].data.new(len(sequences), max_seq_len).fill_(pad_token_id)
        mask = torch.zeros((len(sequences), max_seq_len), device=sequences[0].device)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask


class GPT2Score(SentenceScore):
    """
    Compute the score of a sentence given a GPT-based model.
    Score(sentence) = log( Prob(t_n | t_1 .. t_(n-1)) * ... * Prob(t_1) )
    """

    def _transformer_log_prob(self, sentences_token_ids: List[List[int]]) -> List[float]:
        # Split the sentences into batch of batch_size
        log_prob_scores = []
        for i in range(0, len(sentences_token_ids), self.batch_size):
            batch = sentences_token_ids[i : i + self.batch_size]
            log_prob_scores += self._compute_single_batch(batch)
        return log_prob_scores

    def _compute_single_batch(self, sentences_token_ids: List[List[int]]) -> List[float]:
        # Preprend the context before the sentences to score and add bos special token
        tokens_ids = [
            [self.tokenizer.bos_token_id] + self.context_ids + sentence_token_ids
            for sentence_token_ids in sentences_token_ids
        ]

        # Construct the input tensor
        input_ids, no_pad_mask = self._pad(
            sequences=list(map(lambda ids: torch.tensor(ids, device=self.device), tokens_ids)),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            pred_logits = self.model(input_ids)[0]  # shape = [batch_size, seq_len, vocab_size]
            pred_scores = torch.nn.LogSoftmax(dim=2)(pred_logits)

            # Align input and target
            target_ids = input_ids[:, 1:]
            pred_scores = pred_scores[:, :-1, :]

            # Retrieve the token scores corresponding to the target id
            tokens_scores = pred_scores.gather(dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

            # Zeros the score of pad tokens
            tokens_scores *= no_pad_mask[:, 1:]

        # Return the sum of tokens scores without taking into account the scores of the context tokens
        return torch.sum(tokens_scores[:, len(self.context_ids) :], dim=1).tolist()


class BERTScore(SentenceScore):
    """
    Compute the score of a sentence given a BERT-based model.
    1- mask successively each word of the sentence to score 
        For instance, if the context is "Where is Gael ?" and the sentence to score is "He has left"
        It will create the following mask sentences : 
            - [CLS] Where is Gael ?  [MASK] has left [SEP]
            - [CLS] Where is Gael ? he [MASK] left [SEP]
            - [CLS] Where is Gael ?  he has [MASK] [SEP]
    2- compute the likelihood of each target word that has been mask using context from both side
    3- return the sum all log-likelihood
    """

    def _transformer_log_prob(self, sentences_token_ids: List[List[int]]) -> List[float]:
        """
        1/ Create all the mask_sentences
        2/ Split the mask sentences by batch
            -> One single batch can contain mask_sentences coming from different input sentences
            In order to deal with this issue, the batch will keep for each mask_sentence the following information :
                - mask_sentence_token_ids: list of token ids composing the mask sentence
                - sentence_idx: index of the corresponding input sentence
                - mask_positions: index of the token has have been masked
                - mask_target: token that has been masked
        """
        full_mask_batch = self._add_context_and_generate_mask_sentences(sentences_token_ids)

        mask_log_prob_scores = []
        for i in range(0, len(full_mask_batch), self.batch_size):
            batch = full_mask_batch[i : i + self.batch_size]
            mask_log_prob_scores += self._compute_single_batch(batch)

        # Gather the result for each input sentence
        sentences_log_prob_scores = np.zeros(len(sentences_token_ids))
        for mask_sentence_idx, mask_log_prob_score in enumerate(mask_log_prob_scores):
            sentence_idx = full_mask_batch[mask_sentence_idx]["sentence_idx"]
            sentences_log_prob_scores[sentence_idx] += mask_log_prob_score

        return sentences_log_prob_scores.tolist()

    def _add_context_and_generate_mask_sentences(self, sentences_token_ids: List[List[int]]) -> List[Dict]:
        full_mask_batch = []
        len_context = len(self.context_ids)

        for sentence_idx, sentence_token_ids in enumerate(sentences_token_ids):
            for token_idx, token in enumerate(sentence_token_ids):
                # construct full sentence : [SEP] context sentence [CLS]
                mask_sentence_token_ids = (
                    [self.tokenizer.cls_token_id]
                    + self.context_ids
                    + sentence_token_ids
                    + [self.tokenizer.sep_token_id]
                )
                # replace token n°token_idx by [MASK] token
                mask_sentence_token_ids[1 + len_context + token_idx] = self.tokenizer.mask_token_id

                full_mask_batch.append(
                    {
                        "mask_sentence_token_ids": mask_sentence_token_ids,
                        "sentence_idx": sentence_idx,
                        "mask_positions": 1 + len_context + token_idx,
                        "mask_target": token,
                    }
                )
        return full_mask_batch

    @staticmethod
    def _join_list_of_dict(list_of_dict):
        # for instance, if list_of_dict = [{a:1, b:2}, {a:3, b:4}]
        # will return => {a:[1,3], b:[2,4]}
        return {key: [single_dict[key] for single_dict in list_of_dict] for key in list_of_dict[0].keys()}

    def _compute_single_batch(self, batch: List[Dict]) -> List[float]:
        batch_size = len(batch)
        dict_batch = self._join_list_of_dict(batch)

        input_ids, no_pad_mask = self._pad(
            sequences=list(
                map(lambda ids: torch.tensor(ids, device=self.device), dict_batch["mask_sentence_token_ids"],)
            ),
            pad_token_id=self.tokenizer.sep_token_id,
        )

        with torch.no_grad():
            # contrary to GPT2-based score, we have to provide an attention mask
            # because BERT will also look on the right side and will see the pad tokens
            # by providing no_pad_mask as the attention mask, the model will zero the
            # attention scores of the pad tokens at each layer

            # logits.shape = [batch_size, seq_len, vocab_size]
            logits = self.model(input_ids, attention_mask=no_pad_mask)[0]

            # Retrieve the logits of mask tokens
            # mask_pred_logits.shape = [batch_size, vocac_size]
            mask_pred_logits = logits[range(batch_size), dict_batch["mask_positions"], :]

            # target_score.shape = [batch_size,]
            target_scores = mask_pred_logits[range(batch_size), dict_batch["mask_target"]]
            target_log_probs = target_scores - mask_pred_logits.logsumexp(dim=1)  # from logits to log probs

        return target_log_probs

