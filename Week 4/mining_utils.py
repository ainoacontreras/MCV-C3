
import torch
import torch.nn as nn
import numpy as np

class TripletMarginLossWithMiner(nn.Module):
    def __init__(self, margin, miner, mode="image2text"):
        super(TripletMarginLossWithMiner, self).__init__()
        self.margin = margin
        self.miner = miner
        self.mode = mode

    def forward(self, img_embeddings, text_embeddings):
        img_embeddings = img_embeddings.squeeze()
        text_embeddings = text_embeddings.squeeze()

        # Mine triplets
        triplets = self.miner(img_embeddings, text_embeddings).to(img_embeddings.device)

        # Compute the triplet margin loss
        losses = nn.functional.triplet_margin_loss(
            anchor=img_embeddings[triplets[:, 0]], 
            positive=text_embeddings[triplets[:, 1]], 
            negative=text_embeddings[triplets[:, 2]],
            margin=self.margin
            )
        return losses.mean()

class HardMiner():
    def __init__(self, margin, mode="image2text2"):
        self.margin = margin
        self.mode = mode

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

    def __call__(self, img_embeddings, text_embeddings):
        # Compute the euclidean distance between each image-text pair (distance matrix)
        distance_mat = self.normalized_euclidian_distance(img_embeddings, text_embeddings).cpu()

        triplets = []
        for i in range(img_embeddings.shape[0]):
            # Compute the triplet margin loss for all the possible triplets for the current anchor. If the
            # mode is "image2text", we use the image embeddings as the anchors, and the text embeddings for
            # the positive and negative. Vice versa if the mode is "text2image".
            if self.mode == "image2text":
                margin_loss = distance_mat[i, i] - distance_mat[i] + self.margin
            else:
                margin_loss = distance_mat[i, i] - distance_mat[:, i] + self.margin

            # Treat differently the postive pair
            margin_loss[i] = self.margin - distance_mat[i, i]
            margin_loss = margin_loss.cpu().data.numpy()

            # Ignore triplets that violate the margin
            margin_loss = np.maximum(margin_loss, 0)

            # We sort the negative pairs by its loss value, and take the hardest. If the hardest is
            # also the positive, we take the second hardest.
            sorted_neg_idxs = np.argsort(margin_loss)
            hard_negative = sorted_neg_idxs[-1] if sorted_neg_idxs[-1] != i else sorted_neg_idxs[-2]

            # Append triplet indices (anchor, positive, negative)
            triplets.append([i, i, hard_negative])
        return torch.tensor(triplets)