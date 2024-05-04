import json, itertools, logging, torch
from dainlp.data.utils import use_larger_context


logger = logging.getLogger(__name__)


'''[20220528]'''
class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, filepath, args, tokenizer, label2idx):
        self.args = args
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.documents = [json.loads(l) for l in open(filepath)]
        self.examples = self.convert_documents_to_examples(args.max_span_length, tokenizer)
        self.features = self.convert_examples_to_features(self.examples, self.tokenizer)

    def convert_documents_to_examples(self, max_span_length, tokenizer):
        examples = []
        for document in self.documents:
            sentences, sent_start = [], 0
            for tokens, ner in zip(document["sentences"], document["ner"]):
                ner = [[i[0] - sent_start, i[1] - sent_start, i[2]] for i in ner]
                sentence = {"doc_key": document["doc_key"], "tokens": tokens, "ner": ner}
                outputs = tokenizer(tokens, add_special_tokens=True, padding=False, truncation=False,
                                    is_split_into_words=True)
                input_ids = outputs["input_ids"]
                word_ids = [list(v) for k, v in itertools.groupby([i for i in outputs.word_ids()])]
                assert len(word_ids) == len(tokens) + 2
                start_indices = [sum([len(w) for w in word_ids[0:i + 1]]) for i in range(len(tokens))]
                end_indices = [sum([len(w) for w in word_ids[0:i + 2]]) - 1 for i in range(len(tokens))]

                sentence["input_ids"] = input_ids[1:-1]
                sentence["start_indices"] = [i - 1 for i in start_indices]
                sentence["end_indices"] = [i - 1 for i in end_indices]

                sentence["spans"], sentence["labels"] = [], []
                span2label = {(i[0], i[1]): i[2] for i in ner}
                for start in range(len(tokens)):
                    for end in range(start, min(len(tokens), start + max_span_length)):
                        sentence["spans"].append((start, end, end - start + 1))
                        sentence["labels"].append(span2label.get((start, end), "NA"))
                sent_start += len(tokens)
                sentences.append(sentence)
            if len(sentences) > 0:
                examples.append(sentences)

        if len(examples) > 0:
            logger.info(examples[0])
        return examples

    def convert_examples_to_features(self, examples, tokenizer):
        features = []
        for example in examples:
            all_input_ids, all_spans, all_labels = [], [], []
            for sent in example:
                all_input_ids.append(sent["input_ids"])
                all_spans.append([[sent["start_indices"][s[0]], sent["end_indices"][s[1]], s[2]]
                                  for s in sent["spans"]])
                all_labels.append([self.label2idx[i] for i in sent["labels"]])

            for i in range(len(all_input_ids)):
                input_ids, left_padded, right_padded = use_larger_context(i, all_input_ids, self.args.context_window)
                spans = [[s[0] + left_padded + 1, s[1] + left_padded + 1, s[2]] for s in all_spans[i]]
                if len(input_ids) > self.args.max_seq_length - 2:
                    input_ids = input_ids[0:self.args.max_seq_length - 2]
                    spans = [s for s in spans if s[1] < self.args.max_seq_length - 2]
                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.pad_token_id]
                features.append({"input_ids": input_ids, "spans": spans,
                                 "labels": all_labels[i], "attention_mask": [1] * len(input_ids)})

        if len(features) > 0:
            logger.info(features[0])
            logger.info(tokenizer.convert_ids_to_tokens(features[0]["input_ids"]))

        return features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)


'''[20220528]'''
def test_Dataset(args):
    from transformers import AutoTokenizer
    args.max_span_length = 8
    args.context_window = 300
    tokenizer = AutoTokenizer.from_pretrained("/data/dai031/Corpora/RoBERTa/roberta-base", use_fast=True, add_prefix_space=True)
    label2idx = json.load(open("/data/dai031/ProcessedData/CoNLL2003/0/ner2idx.json"))
    idx2label = {v: k for k, v in label2idx.items()}
    dataset = Dataset("/data/dai031/ProcessedData/CoNLL2003/0/train.json", args, tokenizer, label2idx)

    sentences = [i for e in dataset.examples for i in e]
    for sent, f in zip(sentences, dataset.features):
        golds = []
        for i, (s, l) in enumerate(zip(f["spans"], f["labels"])):
            if l == 0: continue
            span = sent["spans"][i]
            g = [span[0], span[1], idx2label[l]]
            if g not in sent["ner"]:
                print(g)
                print(sent["ner"])
                raise ValueError
            golds.append(g)
        if len(golds) != len(sent["ner"]):
            print(golds)
            print(sent["ner"])


'''[20220524]'''
class Collator:
    pad_label_id = -100

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_seq_length = min(max_seq_length, self.max_seq_length)
        max_num_spans = max([len(f["spans"]) for f in features])
        batch = {"input_ids": [], "attention_mask": [], "spans": [], "labels": [], "span_masks": []}

        for f in features:
            padding_length = max_seq_length - len(f["input_ids"])
            input_ids, attention_mask = f["input_ids"], f["attention_mask"]
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            padding_length = max_num_spans - len(f["spans"])
            spans, labels, span_masks = f["spans"], f["labels"], [1] * len(f["spans"])
            if padding_length > 0:
                spans += [[0, 0, 1]] * padding_length
                labels += [self.pad_label_id] * padding_length
                span_masks += [0] * padding_length
            batch["spans"].append(spans)
            batch["labels"].append(labels)
            batch["span_masks"].append(span_masks)

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch