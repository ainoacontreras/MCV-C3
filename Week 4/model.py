import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import re, fasttext

class Net(torch.nn.Module):
    def __init__(self, text_encoder_type):
        super(Net, self).__init__()

        self.visual_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
        self.visual_encoder.fc = nn.Identity()
        
        self.text_encoder_type = text_encoder_type
        if self.text_encoder_type == 'ft':
            self.text_encoder = fasttext.load_model('fasttext_wiki.en.bin')
            text_dimension = 300
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            self.text_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased")
            self.text_encoder.eval()
            text_dimension = 768

            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(text_dimension, 2048)
        self.relu = nn.ReLU()

    def sentence_to_vector(self, captions, text_type='ft', device=None):
        """
        Convert a sentence into a vector
        """
        if text_type == 'ft':
            batch_embedding = []
            for sentence in captions:
                sentence = sentence.lower()
                words = re.findall(r'\b\w+\b', sentence)
                sentence_embedding = np.stack([
                    self.text_encoder.get_word_vector(word) for word in words if word in self.text_encoder]).mean(axis=0)
                
                batch_embedding.append(torch.tensor(sentence_embedding))
            return torch.stack(batch_embedding).to(device)
        else:
            inputs = self.tokenizer(captions, return_tensors="pt", padding="longest", add_special_tokens=True, return_attention_mask=True).to(device)
            return self.text_encoder(**inputs).last_hidden_state[:, 0, :].to(device)
        
    def normalize_vector(self, vec):
        """print(vec.shape)
        norm = np.sqrt(np.sum(vec**2))
        if not norm==0:
            return vec/norm
        else:
            return vec"""
        return vec / vec.pow(2).sum(dim=1, keepdim=True).sqrt()
        
    def forward(self, image, captions):
        img_embedding = self.normalize_vector(self.visual_encoder(image))
        caption_embedding = self.normalize_vector(
            self.proj(
                self.relu(self.sentence_to_vector(captions, text_type=self.text_encoder_type, device=image.device))))

        return img_embedding, caption_embedding