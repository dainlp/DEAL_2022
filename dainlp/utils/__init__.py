import logging, numpy, random, torch


logger = logging.getLogger(__name__)


'''[Feb-17-2022] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L50'''
def set_seed(seed=52):
    """Fix the random seed for reproducibility"""
    if seed < 0: return
    # os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cuda.matmul.allow_tf32 = False