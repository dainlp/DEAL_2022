import torch


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L166'''
class BertEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings, type_vocab_size,
                 layer_norm_eps, hidden_dropout_prob, position_embedding_type="absolute"):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

        if position_embedding_type != "absolute":
            raise NotImplementedError

        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long,
                                                           device=self.position_ids.device), persistent=False)

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            bs, sq = input_ids.size()
        else:
            bs, sq = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = self.position_ids[:, 0 : sq]

        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :sq].expand(bs, sq)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


'''[Apr-03-2022] 
https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/roberta/modeling_roberta.py#L1559'''
def create_position_ids_from_input_ids(input_ids, padding_idx):
    '''Replace non-padding symbols with their position numbers'''
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


'''[Apr-03-2022] 
https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/roberta/modeling_roberta.py#L142'''
def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx):
    bs, sq = inputs_embeds.size()[:-1]
    # starting from 1
    position_ids = torch.arange(padding_idx + 1, sq + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
    return position_ids.unsqueeze(0).expand((bs, sq))


'''[Apr-03-2022]'''
def test_create_position_ids_from_inputs_embeds():
    from dainlp.modules.embeddings.bert import create_position_ids_from_inputs_embeds
    inputs_embeds = torch.rand((2, 128, 768))
    position_ids = create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx=0)
    print(position_ids.size())
    print(position_ids)


'''[Apr-03-2022] 
https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/roberta/modeling_roberta.py#L70
https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/longformer/modeling_longformer.py#L444'''
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings, type_vocab_size,
                 layer_norm_eps, hidden_dropout_prob, position_embedding_type="absolute"):
        super(RobertaEmbeddings, self).__init__(vocab_size, hidden_size, pad_token_id, max_position_embeddings,
                                                type_vocab_size, layer_norm_eps, hidden_dropout_prob,
                                                position_embedding_type)
        self.padding_idx = pad_token_id
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size,
                                                      padding_idx=self.padding_idx)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = create_position_ids_from_inputs_embeds(input_ids, self.padding_idx)
        return super(RobertaEmbeddings, self).forward(input_ids, token_type_ids=token_type_ids,
                                                      position_ids=position_ids, inputs_embeds=inputs_embeds)