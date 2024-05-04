import logging, torch
from transformers.optimization import AdamW
from enum import Enum


logger = logging.getLogger(__name__)


'''[Apr-01-2022] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/training_args.py#L73'''
class OptimizerNames(Enum):
    ADAMW_HF = "adamw_hf"
    SGD = "sgd"


'''[Apr-01-2022] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L996'''
def get_parameter_names(model, skipped_types):
    result = []
    for name, child in model.named_children():
        if isinstance(child, tuple(skipped_types)): continue
        result += [f"{name}.{n}" for n in get_parameter_names(child, skipped_types)]
    result += list(model._parameters.keys()) # some parameters may be defined with nn.Parameter so not in any child
    return result


'''[Apr-01-2022] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L844'''
def get_optimizer_cls_and_kwargs(args):
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.optim == OptimizerNames.ADAMW_HF:
        adam_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8}
        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif args.optim == OptimizerNames.SGD:
        sgd_kwargs = {"momentum": 0.9}
        optimizer_cls = torch.optim.SGD
        optimizer_kwargs.update(sgd_kwargs)
    else:
        raise ValueError(args.optim)
    return optimizer_cls, optimizer_kwargs


'''[Apr-01-2022] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L806'''
def create_optimizer(model, args):
    if args.task_learning_rate is not None:
        return get_discriminative_AdamW_optimizer(model, args.learning_rate, args.task_learning_rate)

    decay_paras = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_paras = [n for n in decay_paras if "bias" not in n]
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n in decay_paras], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_paras], "weight_decay": 0.0}]
    optimizer_cls, optimizer_kwargs = get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
    return optimizer


'''[May-29-2022] https://github.com/princeton-nlp/PURE/blob/main/run_entity.py#L206'''
def get_discriminative_AdamW_optimizer(model, bert_lr, task_learning_rate):
    params = list(model.named_parameters())
    grouped_parameters = []
    bert_parameters = [p for n, p in params if "bert" in n]
    grouped_parameters.append({"params": bert_parameters})
    task_parameters = [p for n, p in params if "bert" not in n]
    grouped_parameters.append({"params": task_parameters, "lr": task_learning_rate})
    # logger.info(f"use lr={bert_lr} on encoder modules ({[n for n, _ in params if 'bert' in n]}); "
    #             f"lr={task_learning_rate} on task modules ({[n for n, _ in params if 'bert' not in n]})")
    return AdamW(grouped_parameters, lr=bert_lr)