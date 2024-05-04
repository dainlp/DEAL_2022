import torch
from dainlp.modules.utils import FeedForward, select_hidden_states
from dainlp.modules.seq2seq.roberta import RobertaModel, RobertaPreTrainedModel


'''[20220523]'''
class Model(RobertaPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.hidden_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        head_hidden_dim = 150
        self.width_embeddings = torch.nn.Embedding(config.max_span_length + 1, head_hidden_dim)
        self.classifier = torch.nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + head_hidden_dim, num_layers=2,
                        hidden_dims=head_hidden_dim, activations=torch.nn.ReLU(), dropout=0.2),
            torch.nn.Linear(head_hidden_dim, self.num_labels))
        self.init_weights()

    def forward(self, input_ids, spans, labels=None, attention_mask=None, span_masks=None):
        hidden_states = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.hidden_dropout(hidden_states)
        # spans: [batch_size, num_spans, 3]; 0: left offset; 1: right offset; 2: width
        batch_size, num_spans, _ = spans.size()
        spans_start = spans[:, :, 0].view(batch_size, -1)
        spans_start_embeded = select_hidden_states(hidden_states, spans_start)
        spans_end = spans[:, :, 1].view(batch_size, -1)
        spans_end_embeded = select_hidden_states(hidden_states, spans_end)
        spans_width = spans[:, :, 2].view(batch_size, -1)
        spans_width_embeded = self.width_embeddings(spans_width)

        logits = torch.cat((spans_start_embeded, spans_end_embeded, spans_width_embeded), dim=-1)
        for layer in self.classifier:
            logits = layer(logits)
        if labels is None: return {"logits": logits}

        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        active_loss = span_masks.view(-1) == 1
        active_logits = logits.view(-1, logits.shape[-1])
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels))
        loss = loss_fn(active_logits, active_labels)
        return {"logits": logits, "loss": loss}