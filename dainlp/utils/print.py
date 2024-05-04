import datetime, json, logging, math, numpy, os, sys, time
from collections import defaultdict
from dainlp.utils.files import make_sure_parent_dir_exists


logger = logging.getLogger(__name__)


'''[20220528]'''
def get_mean_std(values, latex=False, std=True):
    if isinstance(values, float):
        values = [values]
        std = False

    if len(values) == 0: return "---"
    if std:
        if latex:
            return "%.1f \\tiny $\pm$ %.1f" % (numpy.mean(values), numpy.std(values))
        else:
            return f"{numpy.mean(values):.1f} ({numpy.std(values):.1f})"
    return f"{numpy.mean(values):.1f}"


'''[20220528]'''
def get_table_cell(v, latex=True, highlight=False):
    if highlight:
        if latex:
            return "\\textbf{%s}" % v
        else:
            return f"**{v}**"
    else:
        return v


'''[20220528]'''
def print_in_table(data, columns=[], rows=[], latex=True, highlight={}):
    for r in rows:
        values = []
        for c in columns:
            v = data.get((r, c), c)
            values.append(get_table_cell(v, latex, highlight.get((r, c), False)))

        if latex:
            print(f"{r} & {' & '.join(values)} \\\\ ")
        else:
            print(f"| {r} | {' | '.join(values)} | ")


'''[20220401]'''
def estimate_remaining_time(current_step, total_step, start_time):
    ratio = float(current_step) / total_step
    elapsed = time.time() - start_time
    if current_step == 0: return 0, elapsed, 0

    remaining = elapsed * (1 - ratio) / ratio
    return ratio * 100, elapsed, remaining


'''[20220401]'''
def log_remaining_time(current_step, total_step, start_time, prefix="", suffix=""):
    ratio, elapsed, remaining = estimate_remaining_time(current_step, total_step, start_time)
    logger.info(f"{prefix}Progress: {current_step}/{total_step} ({ratio:.1f}%); "
                f"Elapsed: {print_seconds(elapsed)}; Estimated remaining: {print_seconds(remaining)}{suffix}")


'''[20220401]'''
def print_large_integer(number, suffix=None):
    if suffix is None:
        if number < 1e3: return f"{number}"
        str_number = str(number)
        if number < 1e6: return f"{str_number[:-3]},{str_number[-3:]}"
        if number < 1e9: return f"{str_number[:-6]},{str_number[-6:-3]},{str_number[-3:]}"
        raise ValueError(f"Cannot display ({number})")
    else:
        if suffix == "B": return f"{float(number)/1e9:.1f}B"
        if suffix == "M": return f"{float(number)/1e6:.1f}M"
        if suffix == "K": return f"{float(number)/1e3:.1f}K"

        if number < 1e3: return f"{number}"
        if number < 1e6: return f"{float(number)/1e3:.1f}K"
        if number < 1e9: return f"{float(number)/1e6:.1f}M"
        return f"{float(number)/1e9:.1f}B"


'''[20220330] 
11111 -> 3:05:11.00'''
def print_seconds(seconds):
    msec = int(abs(seconds - int(seconds)) * 100)
    return f"{datetime.timedelta(seconds=int(seconds))}.{msec:02d}"


'''[20220401]'''
def print_delta(values, baselines):
    return f"{numpy.mean(values) - numpy.mean(baselines):.1f}"


'''[20220528]'''
def analyse_test_results(test_metric_dir):
    all_test_results = defaultdict(list)
    for filename in os.listdir(test_metric_dir):
        sp = filename[0:filename.rfind("seed") - 1].split("_")
        test_metric = json.load(open(os.path.join(test_metric_dir, filename)))["test_metrics"]
        all_test_results[tuple([sp[i] for i in range(0, len(sp), 2)])].append(test_metric)
    return all_test_results


'''[20220502]'''
def analyse_dev_and_test_results(train_metric_filepaths, test_metric_filepaths, sort_metric_name, hp_names):
    if isinstance(hp_names, list): hp_names = {i: None for i in hp_names}
    all_dev_results, all_test_results = defaultdict(list), defaultdict(list)
    if test_metric_filepaths is not None:
        assert len(train_metric_filepaths) == len(test_metric_filepaths)

    for i, filepath in enumerate(train_metric_filepaths):
        train_metric = json.load(open(filepath))
        hp_values, expected_hp_values = [], []
        for hp, expected in hp_names.items():
            hp_values.append(train_metric["args"][hp])
            if expected is not None:
                expected_hp_values.append(expected)
            else:
                expected_hp_values.append(train_metric["args"][hp])
        hp_values = tuple(hp_values)
        expected_hp_values = tuple(expected_hp_values)
        if hp_values != expected_hp_values: continue
        all_dev_results[hp_values].append(train_metric["dev_metrics"])
        if test_metric_filepaths is not None and os.path.exists(test_metric_filepaths[i]):
            test_metric = json.load(open(test_metric_filepaths[i]))
            if "test_metrics" in test_metric: test_metric = test_metric["test_metrics"]
            all_test_results[hp_values].append(test_metric)

    assert len(all_dev_results) > 0
    sorted_dev_results = sorted(all_dev_results.items(),
                                key=lambda kv: numpy.mean([v[sort_metric_name] for v in kv[1]]), reverse=True)
    best_hp = sorted_dev_results[0][0]
    best_dev_results = all_dev_results[best_hp]
    test_results = None if test_metric_filepaths is None else all_test_results[best_hp]
    return {"best_dev_results": best_dev_results, "test_results": test_results,
            "best_hp": best_hp, "all_dev_results": all_dev_results, "all_test_results": all_test_results}


'''[20220401]'''
def test_analyse_dev_and_test_results():
    from dainlp.utils.print import analyse_dev_and_test_results

    train_metric_filepaths, test_metric_filepaths = [], []
    dataset_name = "ddi2013"
    for dir, _, filenames in os.walk("/home/dai031/Desktop/Experiments/x-roberta-rel-220317A/results/8"):
        for f in filenames:
            if not f.startswith(dataset_name): continue
            if dir.endswith("train"):
                train_metric_filepaths.append(os.path.join(dir, f))
            else:
                assert dir.endswith("test")
                test_metric_filepaths.append(os.path.join(dir, f))
    results = analyse_dev_and_test_results(train_metric_filepaths, test_metric_filepaths,
                                           "dev_f1", ["learning_rate"])
    print(results["best_dev_results"])
    print(results["test_results"])
    print(results["best_hp"])


'''[May-14=2022] https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/trainer_pt_utils.py#L641
https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/trainer_pt_utils.py#L615'''
def log_metrics(split, metrics, is_world_process_zero=True):
    if not is_world_process_zero: return

    logger.info(f"***** {split} metrics *****")
    metrics_formatted = metrics.copy()
    for k, v in metrics_formatted.items():
        if "_memory_" in k:
            metrics_formatted[k] = f"{v >> 20}MB"
        elif k.endswith("_runtime"):
            metrics_formatted[k] = print_seconds(v)
        elif k == "total_flos":
            metrics_formatted[k] = f"{int(v) >> 30}GF"
        elif isinstance(metrics_formatted[k], float):
            metrics_formatted[k] = round(v, 4)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for k in sorted(metrics_formatted.keys()):
        logger.info(f"  {k: <{k_width}} = {metrics_formatted[k]:>{v_width}}")


'''[20220217]'''
def print_memory_size(size):
    size = size >> 10
    if size < 1024: return f"{size:.1f}K"
    size = size >> 10
    if size < 1024: return f"{size:.1f}M"
    size = size >> 10
    return f"{size:.1f}G"


'''[20220510]'''
def test_print_memory_size():
    size = 1000000
    print(print_memory_size(size))
    print(size / math.pow(2, 20))


'''[20220514]'''
def set_logging_format(log_filepath=None, debug=False):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_filepath is not None:
        make_sure_parent_dir_exists(log_filepath)
        handlers.append(logging.FileHandler(filename=log_filepath))

    if debug:
        logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.DEBUG, handlers=handlers)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=handlers)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L257'''
def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        result[f"{split}_samples_per_second"] = round(num_samples / runtime, 3)
    if num_steps is not None:
        result[f"{split}_steps_per_second"] = round(num_steps / runtime, 3)
    return result