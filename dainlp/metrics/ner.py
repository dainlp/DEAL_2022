import logging, numpy as np
from dainlp.data.ner.utils import tags_to_spans
from dainlp.metrics.utils import F_Score, calculate_f1


logger = logging.getLogger(__name__)


'''[20220520]'''
def score_spans(all_pred_spans, all_gold_spans):
    f_score = F_Score()
    for pred_spans, gold_spans in zip(all_pred_spans, all_gold_spans):
        for p in pred_spans:
            if p in gold_spans:
                f_score.add_tp(p[-1])
            else:
                f_score.add_fp(p[-1])
        for g in gold_spans:
            if g not in pred_spans:
                f_score.add_fn(g[-1])
    return f_score.get_detailed_scores()


'''[20220528]'''
class SpanMetric:
    def __init__(self, idx2label, examples):
        self.idx2label = idx2label
        self.examples = examples

    @staticmethod
    def get_labels_from_logitis(all_logits, all_golds, idx2label, examples):
        sentences = [s for e in examples for s in e]
        all_preds = np.argmax(all_logits, axis=-1)
        assert len(all_preds) == len(sentences)

        all_pred_spans, all_gold_spans = [], []
        for i in range(len(all_preds)):
            all_pred_spans.append([])
            all_gold_spans.append(sentences[i]["ner"])
            for j in range(len(all_preds[i])):
                if all_golds[i][j] == -100:
                    continue
                if all_preds[i][j] != 0:
                    span = sentences[i]["spans"][j]
                    all_pred_spans[-1].append([span[0], span[1], idx2label[all_preds[i][j]]])
        return all_pred_spans, all_gold_spans

    def __call__(self, all_logits, all_golds):
        all_pred_spans, all_gold_spans = SpanMetric.get_labels_from_logitis(all_logits, all_golds,
                                                                            self.idx2label, self.examples)
        return score_spans(all_pred_spans, all_gold_spans)


'''[20220520]'''
class TaggingMetric:
    def __init__(self, idx2label):
        self.idx2label = idx2label

    def __call__(self, all_logits, all_golds):
        all_pred_tags, all_gold_tags = TaggingMetric.get_tags_from_logitis(all_logits, all_golds, self.idx2label)
        all_pred_spans = [tags_to_spans(i) for i in all_pred_tags]
        all_gold_spans = [tags_to_spans(i) for i in all_gold_tags]
        return score_spans(all_pred_spans, all_gold_spans)

    @staticmethod
    def get_tags_from_logitis(all_logits, all_golds, idx2label):
        all_preds = np.argmax(all_logits, axis=-1)
        all_pred_tags, all_gold_tags = [], []
        for preds, golds in zip(all_preds, all_golds):
            pred_tags, gold_tags = [], []
            for p, g in zip(preds, golds):
                if g != -100:
                    pred_tags.append(idx2label[p])
                    gold_tags.append(idx2label[g])
            all_pred_tags.append(pred_tags)
            all_gold_tags.append(gold_tags)
        return all_pred_tags, all_gold_tags


'''[20220520]'''
METRICS = {"tagging": TaggingMetric, "span-ner": SpanMetric}