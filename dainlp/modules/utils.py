import inspect, logging, math, os, re, torch
from dainlp.utils.tensors import debug_tensor


logger = logging.getLogger(__name__)


'''[Apr-01-2022] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L1730'''
def retrieve_modules_from_names(model, names, add_prefix=False, remove_prefix=False):
    module_keys = set([".".join(n.split(".")[:-1]) for n in names])
    # torch.nn.ParameterList is a special case where two parameter keywords
    # are appended to the module name, *e.g.* bert.special_embeddings.0
    module_keys = module_keys.union(set([".".join(n.split(".")[:-2]) for n in names if n[-1].isdigit()]))

    retrieved_modules = []
    # retrieve all modules that has at least one missing weight name
    for name, module in model.named_modules():
        if remove_prefix:
            name = ".".join(name.split(".")[1:]) if name.startswith(model.base_model_prefix) else name
        elif add_prefix:
            name = ".".join([model.base_model_prefix, name]) if len(name) > 0 else model.base_model_prefix

        if name in module_keys:
            retrieved_modules.append(module)
    return retrieved_modules


'''[Mar-30-2022] https://github.com/pytorch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py#L36'''
def get_sinusoidal_embeddings(num_embeddings, embedding_dim, padding_idx=None):
    assert embedding_dim % 2 == 0
    emb = math.log(10000) / (embedding_dim // 2 - 1)
    emb = torch.exp(torch.arange(embedding_dim // 2, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


'''[20211102]'''
def select_hidden_states(hidden_states, indicies):
    # given a tensor of shape (batch size, sequence length, embedding_size)
    # choose hidden states corresponding to several positions
    bs, _, dim = hidden_states.size()
    assert bs == indicies.size(0)
    sq = indicies.size(1)
    select_indices = indicies.unsqueeze(-1).expand(bs, sq, dim)
    hidden_states = hidden_states.gather(1, select_indices)
    return hidden_states # (bs, sq, dim)
    # alternative solution : https://github.com/allenai/allennlp/blob/v2.8.0/allennlp/nn/util.py#L1301
    # batch_idx = torch.arange(0, bs)
    # batch_idx = torch.repeat_interleave(batch_idx, sq)
    # selected_hidden_sates = hidden_states[batch_idx, indices.reshape(-1)]
    # selected_hidden_sates = selected_hidden_sates.reshape((bs, sq, =1))


'''[20220423]'''
def extract_span_embeddings(hidden_states, mask, pooling="max"):
    if pooling == "max":
        extended_mask = mask.unsqueeze(-1).expand(hidden_states.size())
        extended_mask = torch.mul(hidden_states, extended_mask)
        extended_mask = torch.max(extended_mask, dim=1)[0]
    else:
        extended_mask = mask.unsqueeze(1)
        extended_mask = torch.bmm(extended_mask.float(), hidden_states)
        extended_mask = extended_mask.squeeze(1)
        if pooling == "avg":
            extended_mask /= mask.sum(dim=-1).unsqueeze(1)
    return extended_mask


'''[20220423]'''
def test_extract_span_embeddings():
    hidden_states = torch.rand((3, 5, 2)).cuda()
    debug_tensor(hidden_states)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0]]).cuda()
    span_embeddings = extract_span_embeddings(hidden_states, mask)
    print(span_embeddings)
    span_embeddings = extract_span_embeddings(hidden_states, mask, pooling="sum")
    print(span_embeddings)
    span_embeddings = extract_span_embeddings(hidden_states, mask, pooling="avg")
    print(span_embeddings)


'''[Dec-29-2021] https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py#L192'''
def copy_weights_from_numpy(weights, module, name, w_shape=None, w_t=False, b_shape=None, b_t=False, layer_norm=False):
    w = torch.from_numpy(weights[os.path.join(name, "scale" if layer_norm else "kernel")])
    if w_shape is not None:
        w = w.view(w_shape)
    if w_t:
        w = w.t()
    module.weight.copy_(w)
    b = torch.from_numpy(weights[os.path.join(name, "bias")])
    if b_shape is not None:
        b = b.view(b_shape)
    if b_t:
        b = b.t()
    module.bias.copy_(b)


'''[Apr-21-2022] https://github.com/allenai/allennlp/blob/v2.8.0/allennlp/modules/feedforward.py#L13'''
class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout=0.0):
        super(FeedForward, self).__init__()
        hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        activations = activations if isinstance(activations, list) else [activations] * num_layers
        dropout = dropout if isinstance(dropout, list) else [dropout] * num_layers
        assert len(hidden_dims) == num_layers and len(activations) == num_layers and len(dropout) == num_layers

        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        self._linear_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(input_dims, hidden_dims)])
        self._dropout = torch.nn.ModuleList([torch.nn.Dropout(p=v) for v in dropout])

    def forward(self, inputs):
        outputs = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            outputs = dropout(activation(layer(outputs)))
        return outputs


'''[Apr-01-2022] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L1558'''
def load_state_dict_into_model(model, state_dict, base_model_prefix,
                               keys_to_ignore_on_load_missing=None, keys_to_ignore_on_load_unexpected=None):
    old_keys, new_keys = [], []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key: new_key = key.replace("gamma", "weight")
        if "beta" in key: new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    expected_keys, loaded_keys = list(model.state_dict().keys()), list(state_dict.keys())
    prefix = model.base_model_prefix

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module, expects_prefix_module = False, False
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module  # copy model w/o prefix to model w/
    add_prefix_to_model = has_prefix_module and not expects_prefix_module  # copy model w/ prefix to model w/o
    if remove_prefix_from_model:
        expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(prefix)]
        expected_keys = [".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys]
    elif add_prefix_to_model:
        expected_keys = [".".join([prefix, s]) for s in expected_keys]

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))
    if keys_to_ignore_on_load_missing is not None:
        for pat in keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    if keys_to_ignore_on_load_unexpected is not None:
        for pat in keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    unintialized = retrieve_modules_from_names(model, missing_keys, add_prefix_to_model, remove_prefix_from_model)
    for module in unintialized:
        model._init_weights(module)

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None: state_dict._metadata = metadata

    error_msgs = []
    # PyTorch's _load_from_state_dict does not copy parameters in a module's descendants, so need this recursion
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, [], [], error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_prefix = ""
    model_to_load = model
    if len(base_model_prefix) > 0 and not hasattr(model, base_model_prefix) and has_prefix_module:
        start_prefix = base_model_prefix + "."
    if len(base_model_prefix) > 0 and hasattr(model, base_model_prefix) and not has_prefix_module:
        model_to_load = getattr(model, base_model_prefix)
        if any(key in expected_keys_not_prefixed for key in loaded_keys):
            raise ValueError(expected_keys_not_prefixed)
    load(model_to_load, prefix=start_prefix)

    assert len(error_msgs) == 0
    if len(unexpected_keys) > 0: logger.info("\n\t".join(["Unexpected keys:"] + unexpected_keys))
    if len(missing_keys) > 0: logger.info("\n\t".join(["Missing keys:"] + missing_keys))
    return model


'''[TODO] https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L2856'''
def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    assert len(inspect.signature(forward_fn).parameters) == len(input_tensors)

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            assert input_tensor.shape[chunk_dim] == tensor_shape
        assert input_tensors[0].shape[chunk_dim] % chunk_size == 0
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)
    return forward_fn(*input_tensors)