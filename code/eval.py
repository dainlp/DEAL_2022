import dataclasses, json, logging, os, sys, time
sys.path.insert(0, "../")
import dainlp
from dainlp.utils.args import HfArgumentParser, ArgumentsForSpanNER as Arguments
from dainlp.data.ner.span import Dataset, Collator
from dainlp.models.ner.span import Model
from dainlp.metrics.ner import METRICS
from dainlp.training import Trainer
from dainlp.utils.print import print_seconds
from transformers import AutoTokenizer, AutoConfig


logger = logging.getLogger(__name__)


def parse_args():
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    args._setup_devices
    dainlp.utils.print.set_logging_format(os.path.join(args.output_dir, "eval.log"), args.debug)
    dainlp.utils.set_seed(args.seed)
    logger.info(f"DaiNLP {dainlp.__version__}")
    logger.info(args)
    return args


def load_data(args):
    logger.info("**************************************************")
    logger.info("*               Load the datasets                *")
    logger.info("**************************************************")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, add_prefix_space=True)
    idx2label = json.load(open(os.path.join(args.model_dir, "config.json")))["id2label"]
    idx2label = {int(k): v for k, v in idx2label.items()}
    label2idx = {v: k for k, v in idx2label.items()}
    test_dataset = Dataset(args.test_filepath, args, tokenizer, label2idx=label2idx)
    return tokenizer, test_dataset, idx2label


def build_trainer(tokenizer, args, test_dataset, idx2label):
    logger.info("**************************************************")
    logger.info("*               Build the trainer                *")
    logger.info("**************************************************")
    config = AutoConfig.from_pretrained(args.model_dir)
    model = Model.from_pretrained(args.model_dir, config=config)
    data_collator = Collator(tokenizer, args.max_seq_length)
    compute_metrics = METRICS.get(args.dataset_name, METRICS[args.task_name])(idx2label, test_dataset.examples)
    trainer = Trainer(model=model, args=args, data_collator=data_collator, compute_metrics=compute_metrics)
    return trainer, compute_metrics


def main():
    args = parse_args()
    tokenizer, test_dataset, idx2label = load_data(args)
    trainer, compute_metrics = build_trainer(tokenizer, args, test_dataset, idx2label)
    test_outputs = trainer.predict(test_dataset, metric_key_prefix="test")

    if args.output_predictions_filepath is not None:
        preds, _ = compute_metrics.get_labels_from_logitis(test_outputs["logits"], test_outputs["golds"],
                                                           idx2label, test_dataset.examples)
        dainlp.utils.files.write_object_to_json_file(preds, args.output_predictions_filepath)

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)
    dainlp.utils.files.write_object_to_json_file(
        {"args": dataclasses.asdict(args), "test_metrics": test_outputs["metrics"]}, args.output_metrics_filepath)


if __name__ == "__main__":
    main()