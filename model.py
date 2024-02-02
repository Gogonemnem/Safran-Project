import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class AeroBOTSequenceClassification(nn.Module):
    def __init__(self, encoder_name, num_labels):
        super(AeroBOTSequenceClassification, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dense = nn.Linear(self.encoder.config.hidden_size, 32)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(32, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token
        x = self.dense(cls_token)
        x = self.relu(x)
        logits = self.classifier(x)
        return logits

class SequenceClassificationModel(torch.nn.Module):
    def __init__(self, encoder_name='bert-base-uncased', num_labels=15, model_name=None):
        super(SequenceClassificationModel, self).__init__()
        self.original_name = encoder_name
        self.model_name = encoder_name.replace("/", "_")
        if model_name == "AeroBOT":
            self.l1 = AeroBOTSequenceClassification(encoder_name, num_labels)
        else:
            self.l1 = AutoModelForSequenceClassification.from_pretrained(encoder_name, num_labels=num_labels)

    def forward(self, data):
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        if isinstance(self.l1, AeroBOTSequenceClassification):
            return output  # Custom model returns logits directly
        else:
            return output.logits
    
    @staticmethod
    def get_tokenizer(name):
        return AutoTokenizer.from_pretrained(name)
    
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.original_name)
    
    def _set_layer_trainable(self, layer, trainable):
        for param in layer.parameters():
            param.requires_grad = trainable

    def _find_and_set_encoder_layers(self, module, layer_nums, trainable):
        found = False
        # Check if module has 'encoder' and 'layer' attributes directly
        if hasattr(module, 'encoder') and hasattr(module.encoder, 'layer'):
            for layer_num in layer_nums:
                try:
                    layer = module.encoder.layer[layer_num]
                    self._set_layer_trainable(layer, trainable)
                    found = True
                except IndexError:
                    print(f"Layer {layer_num} not found in the encoder.")
        else:
            # Recursively search for encoder in child modules if direct access failed
            for child in module.children():
                if self._find_and_set_encoder_layers(child, layer_nums, trainable):
                    found = True
                    break  # Found and set, no need to continue searching
        return found

    def set_trainable_layers(self, layer_nums=None):
        if layer_nums is not None:
            self.model_name = f'{self.model_name}_Unfrozen{layer_nums}'
            
        # Freeze all parameters first
        self._set_layer_trainable(self, False)  # Freeze everything first

        # Unfreeze specified encoder layers
        if layer_nums:
            if not self._find_and_set_encoder_layers(self.l1, layer_nums, True):
                print("Specified encoder layers not found or unfrozen.")

        # Unfreeze custom layers if present
        if hasattr(self.l1, 'classifier'):
            self._set_layer_trainable(self.l1.classifier, True)
        if hasattr(self.l1, 'dense'):
            self._set_layer_trainable(self.l1.dense, True)