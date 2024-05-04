import logging


logger = logging.getLogger(__name__)


'''[20220508]'''
def bioes_to_bio(tags):
    bio_tags = []
    for i, tag in enumerate(tags):
        if tag == "O" or tag.startswith("B") or tag.startswith("I"):
            bio_tags.append(tag)
        elif tag.startswith("S"):
            bio_tags.append(f"B-{tag[2:]}")
        elif tag.startswith("E"):
            bio_tags.append(f"I-{tag[2:]}")
        else:
            raise ValueError(f"Invalid tag: {tag}")
    return bio_tags


'''[20220508]'''
def bio_to_bioes(tags):
    bioes_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            bioes_tags.append(tag)
        elif tag.startswith("B"):
            if i == len(tags) - 1 or tags[i + 1] != f"I-{tag[2:]}":
                bioes_tags.append(f"S-{tag[2:]}")
            else:
                bioes_tags.append(tag)
        elif tag.startswith("I"):
            if i == len(tags) - 1 or tags[i + 1] != f"I-{tag[2:]}":
                bioes_tags.append(f"E-{tag[2:]}")
            else:
                bioes_tags.append(tag)
        else:
            raise ValueError(f"Invalid tag: {tag}")
    return bioes_tags


'''[May-08-2022] https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py#L145'''
def tags_to_spans(tags):
    def _validate_tag(tag):
        if tag in ["O", "B", "I", "E", "S"]: return
        assert tag.startswith(("B-", "I-", "E-", "S-"))

    def _start_of_span(prev_tag, cur_tag, prev_type, cur_type):
        # current position indicates the beginning of a new mention
        if cur_tag in ["B", "S"]: return True
        if prev_tag == "E" and cur_tag in ["E", "I"]: return True
        if prev_tag == "S" and cur_tag in ["E", "I"]: return True
        if prev_tag == "O" and cur_tag in ["E", "I"]: return True
        if cur_tag != "O" and prev_type != cur_type: return True
        return False

    def _end_of_span(prev_tag, cur_tag, prev_type, cur_type):
        # previous position indicates the end of a mention
        if prev_tag in ["E", "S"]: return True
        if prev_tag == "B" and cur_tag in ["B", "S", "O"]: return True
        if prev_tag == "I" and cur_tag in ["B", "S", "O"]: return True
        if prev_tag != "O" and prev_type != cur_type: return True
        return False

    # for nested list
    # if any(isinstance(s, list) for s in tags): tags = [i for s in tags for i in s + ["O"]]

    spans = []
    prev_tag, prev_type, start = "O", "", 0
    for i, tag in enumerate(tags + ["O"]):
        _validate_tag(tag)
        cur_tag, cur_type = ("O", "") if tag == "O" else tag.split("-")
        if _end_of_span(prev_tag, cur_tag, prev_type, cur_type):
            spans.append([start, i - 1, prev_type])
        if _start_of_span(prev_tag, cur_tag, prev_type, cur_type):
            start = i
        prev_tag, prev_type = cur_tag, cur_type
    return spans


'''[20220508]'''
def spans_to_bio(spans, seq_len):
    tags = ["O"] * seq_len
    for span in spans:
        start, end, entity_type = span[0], span[1], span[2]
        assert tags[start] == "O"
        tags[start] = f"B-{entity_type}"
        for i in range(start + 1, end + 1):
            assert tags[i] == "O"
            tags[i] = f"I-{entity_type}"
    return tags


'''[20220610]'''
def split_complex_categories(mentions):
    # Outermost mentions: are not included by any other mentions
    # Innermost mentions: do not include any other mentions
    # Multi-type mentions: one span has multiple labels
    # Middle mentions: all other mentions
    outermost, innermost, multitype = [1] * len(mentions), [1] * len(mentions), [0] * len(mentions)
    for i in range(len(mentions)):
        for j in range(len(mentions)):
            if i == j: continue
            if mentions[i][0] <= mentions[j][0] and mentions[j][1] <= mentions[i][1]:
                outermost[j], innermost[i] = 0, 0
                if mentions[i][0] == mentions[j][0] and mentions[j][1] == mentions[i][1]:
                    multitype[i], multitype[j] = 1, 1
    middle = [1 if outermost[i] + innermost[i] + multitype[i] == 0 else 0 for i in range(len(mentions))]
    return outermost, innermost, multitype, middle


'''[20220610]'''
def get_nested_pairs(mentions):
    # Extract all (outer, inner) pairs
    pairs = []
    for i in range(len(mentions)):
        for j in range(len(mentions)):
            if i == j: continue
            if mentions[i][0] <= mentions[j][0] and mentions[j][1] <= mentions[i][1]:
                if mentions[i][0] == mentions[j][0] and mentions[j][1] == mentions[i][1]: continue
                pairs.append(mentions[i] + mentions[j])
    return pairs