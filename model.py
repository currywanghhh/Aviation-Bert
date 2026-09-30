from transformers import BertForSequenceClassification
import torch.nn as nn

class WeightedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, *args, class_weights):
        super().__init__(*args)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        if labels is not None:
            loss = self.loss_fct(outputs.loss, labels)
            outputs = (loss,) + outputs[1:]
        return outputs