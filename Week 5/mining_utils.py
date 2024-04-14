import torch
import torch.nn as nn
import numpy as np
from generation_utils import generate_images, generate_similar_prompt


class TripletMarginLossWithMiner(nn.Module):
    def __init__(self, margin, miner, mode="image2text"):
        super(TripletMarginLossWithMiner, self).__init__()
        self.margin = margin
        self.miner = miner
        self.mode = mode
    
    def forward(self, img_embeddings, text_embeddings, captions=None):
        # Mine triplets
        anchors, positives, negatives, generated_negs = self.miner(img_embeddings, text_embeddings, captions)

        # Compute the triplet margin loss
        losses = nn.functional.triplet_margin_loss(
            anchor=anchors, 
            positive=positives, 
            negative=negatives,
            margin=self.margin
            )
        return losses.mean(), generated_negs
    
class HardMinerWithDA():
    def __init__(self, margin, model, transforms=None, mode="text2image", tokenizer=None, pipeline=None, lm_model=None, max_trials=3):
        self.margin = margin
        self.model = model
        self.transforms = transforms
        # Set retrieval mode
        assert mode == "text2image", "Only implemented for Text-To-Image."
        self.mode = mode
        # Data Augmentation parameters
        self.lm_model = lm_model
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.max_trials = max_trials

    def normalized_euclidian_distance(self, u, v):
        # Compute the pairwise euclidean distance matrix
        dot_product = torch.mm(v, u.t())
        sum_of_squares1 = v.pow(2).sum(dim=1).view(-1, 1)
        sum_of_squares2 = u.pow(2).sum(dim=1).view(1, -1)
        distances = sum_of_squares1 + sum_of_squares2 - 2 * dot_product

        # Take the absolute value of the distances
        distances = torch.abs(distances)

        # Normalize to be in the [0, 1] range
        distances = distances / distances.max(dim=1, keepdim=True)[0]
        return distances

    def generate_hard_embeddings(self, captions, device=None):
        """
            Given a batch of captions, generate a batch hard negative image embeddings.
        """
        hard_prompts = generate_similar_prompt(self.lm_model, tokenizer=self.tokenizer, captions=captions)
        hard_images = []
        for prompt in hard_prompts:
            img = generate_images(self.pipeline, prompt)
            img = self.transforms(img)
            hard_images.append(img)
        hard_images = torch.stack(hard_images).to(device)
        generated_hard_img_embeddings = self.model.normalize_vector(self.model.visual_encoder(hard_images))
        return generated_hard_img_embeddings

    def __call__(self, img_embeddings, text_embeddings, captions=None):            
        # Compute the euclidean distance between each image-text pair (distance matrix)
        distance_mat = self.normalized_euclidian_distance(img_embeddings, text_embeddings).cpu()

        triplets = []
        # Aux list for storing those bacth examples that require a hard image to be generated
        idx_need_generation = []
        for i in range(text_embeddings.shape[0]):
            margin_loss = distance_mat[i, i] - distance_mat[:, i] + self.margin

            # Margin loss for the positive (margin - dist_pos)
            margin_loss[i] = self.margin - distance_mat[i, i]
            margin_loss = margin_loss.cpu().data.numpy()

            # Ignore triplets that violate the margin
            margin_loss = np.maximum(margin_loss, 0)

            # We sort the negative pairs by its loss value, and take the hardest. If the hardest is
            # also the positive, we take the second hardest.
            sorted_neg_idxs = np.argsort(margin_loss)

            # If the "hardest" embedding is the positive, we need to generate a negative image. During inference,
            # we just take the second hardest embedding
            if sorted_neg_idxs[-1] == i:
                if captions is not None:
                    # We will concatenate the generated hard embeddings with the image embeddings, 
                    # so that is the index that we are assigning here
                    hard_negative = img_embeddings.shape[0]+len(idx_need_generation)
                    idx_need_generation.append((i, len(idx_need_generation)))
                else:
                    hard_negative = sorted_neg_idxs[-2]
            else:
                hard_negative = sorted_neg_idxs[-1]
            # Append triplet indices (anchor, positive, negative)
            triplets.append([i, i, hard_negative])
        triplets = torch.tensor(triplets).to(img_embeddings.device)

        # Logging metric
        generated_negs = len(idx_need_generation)

        if captions is not None and len(idx_need_generation):
            # Given the input captions (anchors), generate hard images
            trial = 0
            while trial < self.max_trials:
                interest_indices = [i for i, _ in idx_need_generation]
                interest_positions = [i for _, i in idx_need_generation]

                hard_img_embeddings = self.generate_hard_embeddings(captions=[captions[i] for i in interest_indices], device=img_embeddings.device)

                pos_dist = self.normalized_euclidian_distance(img_embeddings[interest_indices, :], text_embeddings[interest_indices, :]).diag().cpu()
                neg_dist = self.normalized_euclidian_distance(hard_img_embeddings, text_embeddings[interest_indices, :]).diag().cpu()

                # Check if the generated images are hard or not
                is_hard_embed = pos_dist + self.margin > neg_dist
                
                if trial == 0:
                    # In the first trial, we save all the generated image embeddings
                    generated_hard_img_embeddings = hard_img_embeddings
                else:
                    # In the rest of trials, we save the image embeddings that are hard
                    generated_hard_img_embeddings[[i for i, good in zip(interest_positions, is_hard_embed) if good], :] = hard_img_embeddings[is_hard_embed, :]

                # If we have generated all the necessary hard image embeddings, exit the loop
                if is_hard_embed.sum() == is_hard_embed.shape[0]:
                    break

                # Update indices for the next trial
                idx_need_generation = [i for i, good in zip(idx_need_generation, is_hard_embed) if not good]
                trial += 1

            # generated_hard_img_embeddings = self.generate_hard_embeddings(captions=[captions[i] for i in idx_need_generation], device=img_embeddings.device)
            img_embeddings = torch.cat([img_embeddings, generated_hard_img_embeddings])
        

        # Get anchor, positive and negative embeddings
        anchors = text_embeddings[triplets[:, 0]]
        positives = img_embeddings[triplets[:, 1]]
        negatives = img_embeddings[triplets[:, 2]]

        return anchors, positives, negatives, generated_negs