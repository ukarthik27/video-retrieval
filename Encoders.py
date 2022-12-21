import pip
pip.main(['install', '-q', 'transformers']) # Package for pretrained BERT.
pip.main(['install', '-q', 'timm']) # Package for pretrained Xception.

import numpy as np
import torch
import torchvision
from transformers import BertModel
import timm
from yolov5.models.yoloE import BaseModel, DetectionModel,DetectMultiBackend


class InceptionV3Encoder(torch.nn.Module):
    def __init__(self):
        super(InceptionV3Encoder, self).__init__()

        self.incep3 = torchvision.models.inception_v3(pretrained=True)
        self.incep3.aux_logits = False # We don't use the auxiliary output.
        self.output_size = self.incep3.fc.in_features # in_features=2048.
        self.incep3.fc = torch.nn.Identity() # Deactivate the fc layer. torch.nn.Identity has no parameters.
        for parameter in self.incep3.parameters():
            parameter.requires_grad = False # Freeze all the parameters.
        self.num_parameters = sum([np.prod(params.size()) for params in self.incep3.parameters()])

    def forward(self, x):
        return self.incep3(x)

class XceptionEncoder(torch.nn.Module):
    def __init__(self):
        super(XceptionEncoder, self).__init__()

        self.xception = timm.create_model("xception", pretrained=True)
        self.output_size = self.xception.fc.in_features
        self.xception.fc = torch.nn.Identity()
        for parameter in self.xception.parameters():
            parameter.requires_grad = False
        self.num_parameters = sum([np.prod(params.size()) for params in self.xception.parameters()])

    def forward(self, x):
        return self.xception(x)

class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        self.resnet = torchvision.models.resnet101(pretrained=True, progress=False)
        self.output_size = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False
        self.num_parameters = sum([np.prod(params.size()) for params in self.resnet.parameters()])

    def forward(self, x):
        return self.resnet(x)

class YoloEncoder(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg='yolov5s.yaml', model=None, embed_size=300, cutoff=-1, freeze_yolo_layers=True, freeze_embedding_layers=False):  # yaml, model, number of classes, cutoff index
        super().__init__()
        model = DetectionModel()
        self._from_detection_model(model, cutoff, freeze_yolo_layers, freeze_embedding_layers) if model is not None else self._from_yaml(cfg)
        self.output_size = self.model[-1].cv2.conv.out_channels*10*10
        self.num_parameters = sum([np.prod(params.size()) for params in self.model.parameters()])

    def _from_detection_model(self, model, cutoff=10, freeze_yolo_layers=True, freeze_embedding_layers=False):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]
        self.model = model.model
        self.stride = model.stride

        for layer in model.model:
            layer.requires_grad=False

    def forward(self, x):
        x = self.model(x)
        #x = torch.flatten(x, start_dim=1)
        return x

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None

class BertEncoder(torch.nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=False)
        # With bert-base-uncased, all the input are lowercased before being tokenized.
        self.output_size = self.bert.config.hidden_size # hidden_size=768.
        for parameter in self.bert.parameters():
            parameter.requires_grad = False # Freeze all the parameters.
        self.num_parameters = sum([np.prod(params.size()) for params in self.bert.parameters()])

    def forward(self, input_ids, attention_mask):
        last_hidden, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return last_hidden[:, 0]

class GloVeEncoder(torch.nn.Module):
    # No pre-trained nets are used for GloVe Word2Vec. RNN part is trainable.
    def __init__(self):
        super(GloVeEncoder, self).__init__()

        self.glove = torch.nn.Identity()
        self.output_size = [None, 50] # [max_len, word embedding dimension]
        self.num_parameters = 0

    def forward(self, embeddings):
        self.output_size = [embeddings.shape[1], embeddings.shape[2]]
        return self.glove(embeddings)

def Encoder(name):
    if name == "InceptionV3":
        return InceptionV3Encoder()
    elif name == "Xception":
        return XceptionEncoder()
    elif name == "ResNet":
        return ResNetEncoder()
    elif name == "YOLO":
        return YoloEncoder()
    elif name == "BERT":
        return BertEncoder()
    elif name == "GloVe":
        return GloVeEncoder()
    else:
        raise ValueError(name + " has not been implemented!")
