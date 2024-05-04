import logging, torch
from dainlp.modules.attention import MultiHeadSelfAttention


logger = logging.getLogger(__name__)


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L353'''
class SelfAttentionOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps=None, dropout=0.1):
        super(SelfAttentionOutput, self).__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)
        self.LayerNorm = None if layer_norm_eps is None else torch.nn.LayerNorm(output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L367'''
class BertAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads, attention_dropout, layer_norm_eps, hidden_dropout):
        super(BertAttention, self).__init__()
        self.self = MultiHeadSelfAttention(input_dim, num_heads, attention_dropout)
        self.output = SelfAttentionOutput(input_dim, input_dim, layer_norm_eps, hidden_dropout)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        outputs = self.output(self_outputs, hidden_states)
        return outputs


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L416'''
class BertIntermediate(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BertIntermediate, self).__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        return hidden_states


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L431'''
class BertOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps, dropout):
        super(BertOutput, self).__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)
        self.LayerNorm = torch.nn.LayerNorm(output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L445'''
class TransformerLayer(torch.nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = BertAttention(config.hidden_size, config.num_attention_heads,
                                       config.attention_probs_dropout_prob, config.layer_norm_eps,
                                       config.hidden_dropout_prob)
        self.intermediate = BertIntermediate(config.hidden_size, config.intermediate_size)
        self.output = BertOutput(config.intermediate_size, config.hidden_size, config.layer_norm_eps,
                                       config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask)
        hidden_states = self.intermediate(attention_outputs)
        hidden_states = self.output(hidden_states, attention_outputs)
        return hidden_states


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L530'''
class TransformerLayerList(torch.nn.Module):
    def __init__(self, config):
        super(TransformerLayerList, self).__init__()
        self.config = config
        self.layer = torch.nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, attention_mask, layer_head_mask)
        return hidden_states