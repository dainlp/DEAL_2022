import torch
from dainlp.modules import ModelBase
from dainlp.modules.seq2seq.utils import TransformerLayerList
from dainlp.modules.embeddings.bert import RobertaEmbeddings
from transformers import RobertaConfig


'''[Apr-03-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L584'''
class RobertaPreTrainedModel(ModelBase):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


'''[Apr-03-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L695'''
class RobertaModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config.vocab_size, config.hidden_size, config.pad_token_id,
                                            config.max_position_embeddings, config.type_vocab_size,
                                            config.layer_norm_eps, config.hidden_dropout_prob)
        self.encoder = TransformerLayerList(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None):
        assert not self.config.is_decoder
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        bs, sq = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = self.embeddings.token_type_ids[:, :sq].expand(bs, sq)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        hidden_states = self.embeddings(input_ids, token_type_ids=token_type_ids,
                                        position_ids=position_ids, inputs_embeds=inputs_embeds)
        hidden_states = self.encoder(hidden_states, attention_mask=extended_attention_mask, head_mask=head_mask)
        return hidden_states