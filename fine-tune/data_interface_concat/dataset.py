# coding:utf-8
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from common.utils import shift_tokens_right
from common.mtl_config import get_mtl_config
from data_interface_concat.dep_align import align_example, IGNORE as DEP_IGNORE
import json
import hashlib


def padding_func(features, padding_side="right", pad_token_id=1, key="label", pad_to_multiple_of=1, max_length=None):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
        if max_length is not None:
            max_label_length = min(max_length,
                (max_label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            )
        else:
            max_label_length = (max_label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


class AMRParsingDataSet(Dataset):
    def __init__(
        self, tokenizer, args, model_args
    ):
        super().__init__()
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.src_prefix = args.source_prefix
        self.tgt_prefix = args.target_prefix
        self.cache_dir = model_args.cache_dir
        self.use_speaker_prefix = args.use_speaker_prefix
        self.tokenizer = tokenizer
        self.unified_input = args.unified_input

        self.max_src_length = min(args.max_source_length, self.tokenizer.model_max_length)
        self.max_tgt_length = min(args.max_target_length, self.tokenizer.model_max_length)

        data_files = {}
        if self.train_file is not None:
            data_files["train"] = self.train_file
            
        if self.validation_file is not None:
            data_files["validation"] = self.validation_file
            
        if self.test_file is not None:
            data_files["test"] = self.test_file
        
        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # exit()
        self.datasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py",
            data_files=data_files,
            keep_in_memory=False,
        )
        column_names = self.datasets["train"].column_names
        print("datasets:", self.datasets)
        print("colums:", column_names)

        self.trim_counts = {
            "train": {"token_count": 0, "data_count": 0},
            "valid": {"token_count": 0, "data_count": 0},
            "test": {"token_count": 0, "data_count": 0},
        }

        # ---- MTL (M3/M4): label dependensi dari berkas pendamping, HANYA untuk split train ----
        self.mtl = get_mtl_config(self.train_file) if self.train_file else None
        self.dep_stats = {}
        if self.mtl is not None and "train" in self.datasets:
            self._attach_dep_column()

    def _attach_dep_column(self):
        cfg = self.mtl
        rel2id = {r: i for i, r in enumerate(cfg["labels"])}
        ens = self.datasets["train"]["en"]
        rows = []
        with open(cfg["dep_file"], encoding="utf-8") as f:
            for n, line in enumerate(f):
                d = json.loads(line)
                if n >= len(ens):
                    raise ValueError(f"{cfg['dep_file']} lebih panjang dari data train ({len(ens)} baris)")
                h = hashlib.md5(ens[n].encode("utf-8")).hexdigest()[:12]
                if d["i"] != n or d["h"] != h:
                    raise ValueError(f"{cfg['dep_file']} baris {n} tidak cocok dengan train.jsonl "
                                     f"(i={d['i']}, hash {d['h']} != {h})")
                rows.append([[w[0], w[1], w[2], rel2id.get(w[3], -1)] for w in d["w"]])
        if len(rows) != len(ens):
            raise ValueError(f"{cfg['dep_file']} {len(rows)} baris, train {len(ens)} baris")
        self.datasets["train"] = self.datasets["train"].add_column("dep_w", rows)
        print(f"[MTL] label dependensi dipasang: {len(rows)} kalimat, {cfg['n_rels']} label, "
              f"dari {cfg['dep_file']}")

    def create_tokenize_function(self, dataset_type):
        trim_counts = self.trim_counts[dataset_type]
        def tokenize_function(examples):
            amrs = examples["amr"]
            ids = examples["id"]
            ens = examples["en"]
            # txt = ["id_ID " + id + " en_XX " + (en if dataset_type == "train" else "<mask>") for id, en in zip(ids, ens)]
            txt = ["id_ID " + id + " en_XX " + en for id, en in zip(ids, ens)]
            
            amr_ids = [self.tokenizer.tokenize_amr(itm.split())[:self.max_tgt_length-2] + [self.tokenizer.amr_eos_token_id] for itm in amrs]
            
            raw_txt_ids = self.tokenizer(
                txt, max_length=self.max_src_length, padding=False, truncation=True
            )["input_ids"]
            if self.unified_input:
                for itm in raw_txt_ids:
                    if len(itm) > self.max_src_length-3:
                        trim_counts["token_count"] += len(itm) - (self.max_src_length-3)
                        trim_counts["data_count"] += 1
                txt_ids = [itm[:self.max_src_length-3] + [self.tokenizer.amr_bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.amr_eos_token_id] for itm in raw_txt_ids]
            else:
                txt_ids = raw_txt_ids
            out = {
                "input_ids": txt_ids,
                "labels": amr_ids
            }
            if "dep_w" in examples:
                en_id = self.tokenizer.lang_code_to_id["en_XX"]
                dh, dr = [], []
                for ids_, en, words in zip(txt_ids, ens, examples["dep_w"]):
                    h_, r_ = align_example(ids_, en, words, self.tokenizer, en_id, stats=self.dep_stats)
                    dh.append(h_)
                    dr.append(r_)
                out["dep_heads"] = dh
                out["dep_rels"] = dr
            return out
        
        return tokenize_function


@dataclass
class DataCollatorForAMRParsing:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # MTL: label dependensi dipisah dulu (tokenizer.pad tidak bisa mem-pad kunci lain)
        dep = None
        if "dep_heads" in features[0]:
            dep = ([f.pop("dep_heads") for f in features], [f.pop("dep_rels") for f in features])
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        features["decoder_input_ids"] = shift_tokens_right(
            features["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.amr_bos_token_id,
        )

        batch = {
            "input_ids": features["input_ids"],
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }
        if dep is not None:
            import torch
            T = features["input_ids"].shape[1]
            for key, seqs in zip(("dep_heads", "dep_rels"), dep):
                t = torch.full((len(seqs), T), DEP_IGNORE, dtype=torch.long)
                for b, s in enumerate(seqs):
                    s = list(s)[:T]
                    t[b, : len(s)] = torch.tensor(s, dtype=torch.long)
                batch[key] = t
        return batch
        

class AMR2TextDataSet(Dataset):
    def __init__(
        self, tokenizer, args, model_args
    ):
        super().__init__()
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.src_prefix = args.source_prefix
        self.tgt_prefix = args.target_prefix
        self.cache_dir = model_args.cache_dir
        self.use_speaker_prefix = args.use_speaker_prefix
        self.tokenizer = tokenizer
        self.unified_input = args.unified_input

        self.max_src_length = min(args.max_source_length, self.tokenizer.model_max_length)
        self.max_tgt_length = min(args.max_target_length, self.tokenizer.model_max_length)

        data_files = {}
        if self.train_file is not None:
            data_files["train"] = self.train_file
            
        if self.validation_file is not None:
            data_files["validation"] = self.validation_file
            
        if self.test_file is not None:
            data_files["test"] = self.test_file
        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # exit()
        self.datasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py",
            data_files=data_files,
            keep_in_memory=False,
        )
        column_names = self.datasets["train"].column_names
        print("datasets:", self.datasets)
        print("colums:", column_names)

    def tokenize_function(self, examples):
        src = examples["src"]  # AMR tokens
        tgt = examples["tgt"]  # Text tokens
        if not self.unified_input:
            src_ids = [[self.tokenizer.amr_bos_token_id] + self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 2] + [self.tokenizer.amr_eos_token_id] for itm in src]
        else:
            # [<s>[mask]</s><AMR>xxx</AMR>]
            src_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id] + [self.tokenizer.amr_bos_token_id] + self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length -5] + [self.tokenizer.amr_eos_token_id] for itm in src]
            
        with self.tokenizer.as_target_tokenizer():
            tgt_ids = self.tokenizer(
                tgt, max_length=self.max_tgt_length, padding=False, truncation=True
            )
            tgt_ids["input_ids"] = [
                label[1:] for label in tgt_ids["input_ids"]
            ]
        model_inputs = {}
        model_inputs["input_ids"] = src_ids
        model_inputs["labels"] = tgt_ids["input_ids"]
        return model_inputs


@dataclass
class DataCollatorForAMR2Text:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        features["decoder_input_ids"] = shift_tokens_right(
            features["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.eos_token_id,
        )

        return {
            "input_ids": features["input_ids"],
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }