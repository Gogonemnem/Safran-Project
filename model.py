import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class SequenceClassificationModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=15):
        super(SequenceClassificationModel, self).__init__()
        self.original_name = model_name
        self.model_name = model_name.replace("/", "_")
        self.l1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, data):
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
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
        if hasattr(module, 'encoder'):
            for layer_num in layer_nums:
                try:
                    self._set_layer_trainable(module.encoder.layer[layer_num], trainable)
                except IndexError:
                    print(f"Layer {layer_num} not found in the encoder.")
            return True
        else:
            for child in module.children():
                if self._find_and_set_encoder_layers(child, layer_nums, trainable):
                    return True
        return False

    def set_trainable_layers(self, layer_nums=None):
        if layer_nums is not None:
            self.model_name = f'{self.model_name}_Unfrozen{layer_nums}'
            
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze classifier layers
        if hasattr(self.l1, 'classifier'):
            self._set_layer_trainable(self.l1.classifier, True)

        # Attempt to find and unfreeze encoder layers
        if not self._find_and_set_encoder_layers(self.l1, layer_nums or [], True):
            print("Encoder layers not found in the model.")

