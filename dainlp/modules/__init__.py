import os, torch
from dainlp.modules.utils import load_state_dict_into_model
from transformers import PretrainedConfig


'''[Mar-30-2022] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L121'''
def get_parameter_device(parameter):
    try:
        return next(parameter.parameters()).device
    except:
        raise NotImplementedError


'''[Mar-30-2022] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L136'''
def get_parameter_dtype(parameter):
    try:
        return next(parameter.parameters()).dtype
    except:
        raise NotImplementedError


'''[Apr-02-2022] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L151
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L423'''
class ModuleUtilsMixin:
    def add_memory_hooks(self):
        raise NotImplementedError

    def reset_memory_hooks_state(self):
        raise NotImplementedError

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask):
        raise NotImplementedError

    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask, device):
        raise NotImplementedError

    def get_extended_attention_mask(self, attention_mask):
        # 1 for tokens to attend to, 0 for tokens to ignore
        if attention_mask.dim() == 3:  # [bs, from sequence length, to sequence length]
            extended_attention_mask = attention_mask[:, None, :, :] # make it broadcastable to all heads
        elif attention_mask.dim() == 2:  # [bs, sq]
            assert not self.config.is_decoder
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(attention_mask.shape)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        # 0 for tokens to attend to, -10000.0 for tokens to ignore
        # this makes sense when we add it to the raw scores before the softmax
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers):
        assert head_mask is None
        head_mask = [None] * num_hidden_layers
        return head_mask

    def num_parameters(self, only_trainable=False, exclude_embeddings=False):
        if exclude_embeddings:
            embedding = [f"{n}.weight" for n, m in self.named_modules() if isinstance(m, torch.nn.Embedding)]
            non_embedding = [p for n, p in self.named_parameters() if n not in embedding]
            return sum(p.numel() for p in non_embedding if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def floating_point_ops(self, input_dict, exclude_embeddings=True):
        assert "input_ids" in input_dict
        return 6 * input_dict["input_ids"].numel() * self.num_parameters(exclude_embeddings=exclude_embeddings)


'''[Apr-02-2022] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L151
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L423'''
class ModelBase(torch.nn.Module, ModuleUtilsMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids" # pixel_values for vision models
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    _keys_to_ignore_on_save = None
    is_parallelizable = False

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def post_init(self):
        self.init_weights()

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None

    def _init_weights(self, module):
        raise NotImplementedError

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
            assert not self.config.torchscript
            input_embeddings = self.get_input_embeddings()
            output_embeddings.weight = input_embeddings.weight
            if getattr(output_embeddings, "bias", None) is not None:
                pad_length = output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]
                output_embeddings.bias.data = torch.nn.functional.pad(output_embeddings.bias.data, (0, pad_length))
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings
        assert not getattr(self.config, "is_encoder_decoder", False)
        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def resize_token_embeddings(self, new_num_tokens=None):
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None: return model_embeds
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens
        self.tie_weights()
        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        if new_num_tokens is None:
            new_embeddings = old_embeddings
        else:
            old_num_tokens, embedding_dim = old_embeddings.weight.size()
            if old_num_tokens == new_num_tokens:
                new_embeddings = old_embeddings
            else:
                new_embeddings = torch.nn.Embedding(new_num_tokens, embedding_dim)
                new_embeddings.to(self.device, dtype=old_embeddings.weight.dtype)
                self._init_weights(new_embeddings)
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        self.set_input_embeddings(new_embeddings)
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            raise NotImplemented
        return self.get_input_embeddings()

    def init_weights(self):
        assert not self.config.pruned_heads
        # are the following steps necessary?
        self.apply(self._init_weights)
        self.tie_weights()

    def save_pretrained(self, save_directory, state_dict=None):
        assert not os.path.isfile(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        assert not hasattr(self, "module")
        model_to_save = self
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]
        model_to_save.config.architectures = [model_to_save.__class__.__name__]
        model_to_save.config.save_pretrained(save_directory)
        if state_dict is None: state_dict = model_to_save.state_dict()
        if self._keys_to_ignore_on_save is not None:
            for k in self._keys_to_ignore_on_save:
                if k in state_dict.keys():
                    del state_dict[k]
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_dir, **kwargs):
        assert os.path.isdir(pretrained_dir)  # local_files_only == True
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)

        if isinstance(config, PretrainedConfig):
            model_kwargs = kwargs
        else:
            config_path = config if config is not None else pretrained_dir
            config, model_kwargs = cls.config_class.from_pretrained(config if config is not None else pretrained_dir)

        if state_dict is None:
            state_dict = torch.load(os.path.join(pretrained_dir, "pytorch_model.bin"), map_location="cpu")

        config.name_or_path = pretrained_dir
        model = cls(config, **model_kwargs)
        load_state_dict_into_model(model, state_dict, cls.base_model_prefix,
                                   cls._keys_to_ignore_on_load_missing, cls._keys_to_ignore_on_load_unexpected)
        model.tie_weights()
        model.eval()
        return model
