from typing import List


'''[20220401]'''
def use_larger_context(cur_idx: int, all_sequences: List[int], context_window: int):
    cur_seq = all_sequences[cur_idx]
    if context_window <= 0: return cur_seq, 0, 0

    left_added, right_added = 0, 0
    left_budget = (context_window - len(cur_seq)) // 2
    right_budget = context_window - len(cur_seq) - left_budget
    prev_idx = cur_idx - 1
    while prev_idx >= 0 and left_budget > 0:
        context_to_add = all_sequences[prev_idx][-left_budget:]
        cur_seq = context_to_add + cur_seq
        left_budget -= len(context_to_add)
        left_added += len(context_to_add)
        prev_idx -= 1

    next_idx = cur_idx + 1
    while next_idx < len(all_sequences) and right_budget > 0:
        context_to_add = all_sequences[next_idx][:right_budget]
        cur_seq = cur_seq + context_to_add
        right_budget -= len(context_to_add)
        right_added += len(context_to_add)
        next_idx += 1

    return cur_seq, left_added, right_added