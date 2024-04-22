import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, BertModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(21632, 2048)
        self.fc2 = nn.Linear(2048, 768)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class AudioModel(nn.Module):
    
    def __init__(self, device):
        super(AudioModel, self).__init__()
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32

        model_id = "distil-whisper/distil-large-v3"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.audio_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2", device_map=device)
        self.audio_encoder = self.audio_encoder.eval()

        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        
        self.device = device
    
    def forward(self, audio):

        # convert batch tensor [batch_size, audio_length] to list of batch np arrays
        audio = [audio[i].cpu().numpy() for i in range(audio.shape[0])]

        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to(self.device, dtype=self.torch_dtype)
        return self.audio_encoder.model.encoder(**input_features).last_hidden_state[:, :input_features['attention_mask'][0].sum(),:].to(torch.float32)
    
class TextModel(nn.Module):

    def __init__(self, device, replace_token=''):
        super(TextModel, self).__init__()
        self.replace_token = replace_token
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.text_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased", device_map=device)
        self.text_encoder = self.text_encoder.eval()
        self.device = device

        for param in self.text_encoder.parameters():
            param.requires_grad = False 
    
    def forward(self, text):
        
        if self.replace_token == '[MASK]':
            """
            If the replacement token is [MASK], we create the embedding by masking each token in the input text and predicting the masked token using the model.
            """
            # Tokenize the modified input text
            inputs = self.tokenizer(text, return_tensors="pt", padding="longest", add_special_tokens=True, return_attention_mask=True).to(self.device)

            # Identify the positions of the [MASK] tokens
            masked_indices = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

            # Forward pass through the model to obtain predictions for the masked tokens
            outputs = self.text_encoder(**inputs).last_hidden_state

            # Retrieve the predicted tokens for the [MASK] positions
            predicted_tokens = torch.argmax(outputs[0, masked_indices], dim=-1)

            # Replace the [MASK] tokens in the input with the predicted tokens
            for i, idx in enumerate(masked_indices):
                inputs["input_ids"][0, idx] = predicted_tokens[i]

        elif self.replace_token == "":
            """
            If the replacement token is an empty string, we create the embedding by ommitting the inaudible parts from the input text.
            """
            inputs = self.tokenizer(text, return_tensors="pt", padding="longest", add_special_tokens=True, return_attention_mask=True).to(self.device)
        else:
            raise ValueError("Invalid replacement token")
        
        return self.text_encoder(**inputs).last_hidden_state[:, 0, :]

class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_length, hidden_dim]
        attention_weights = self.attention(x).squeeze(-1)  # [batch_size, seq_length]
        attention_weights = F.softmax(attention_weights, dim=1)  # softmax over seq_length
        pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, hidden_dim]
        return pooled
    
class MultiModalModel(nn.Module):
    def __init__(self, device, num_classes=7, replace_token='', 
                 hidden_dim_audio=1280, hidden_dim_text=768, hidden_dim_vision=768, 
                 use_audio=True, use_text=True, use_vision=True):
        super(MultiModalModel, self).__init__()
        if use_vision:
            self.vision_model = VisionModel()
        if use_audio:
            self.audio_model = AudioModel(device)
        if use_text:
            self.text_model = TextModel(device, replace_token)

        self.use_audio = use_audio
        self.use_text = use_text
        self.use_vision = use_vision

        self.attentive_pooling = AttentivePooling(hidden_dim_audio)
        self.fc = nn.Linear(hidden_dim_audio*use_audio + hidden_dim_text*use_text + hidden_dim_vision*use_vision, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, image, audio, text):
        embeddings = []
        if self.use_vision:
            vision_output = self.vision_model(image)
            embeddings.append(vision_output)
        
        if self.use_audio:
            with torch.no_grad():
                audio_output = self.audio_model(audio)
            audio_output = self.attentive_pooling(audio_output)
            embeddings.append(audio_output)

        if self.use_text:
            with torch.no_grad():
                text_output = self.text_model(text)
            embeddings.append(text_output)

        x = torch.cat(embeddings, dim=1)
        x = self.dropout(F.relu(self.fc(x)))
        x = self.fc2(x)

        return x