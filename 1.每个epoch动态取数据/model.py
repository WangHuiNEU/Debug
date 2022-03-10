

import torch
import torch.nn as nn


class MLP_layers(torch.nn.Module):
    def __init__(self, layers, drop_rate):
        super(MLP_layers, self).__init__()
        self.activate = nn.GELU()
        self.drop_rate = drop_rate
        self.layers = layers

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, x):
        return self.mlp_layers(x)


class Model(torch.nn.Module):

    def __init__(self, args, user_num, item_num):
        super(Model, self).__init__()
        self.args = args
        self.user_embedding = nn.Embedding(user_num, args.embedding_dim)
        self.id_embedding = nn.Embedding(item_num, args.embedding_dim)
        # [args.embedding_dim]*4
        self.user_encoder = MLP_layers(layers=[args.embedding_dim, 512, 256, 128],
                                       drop_rate=args.drop_rate)
        self.item_encoder = MLP_layers(layers=[args.embedding_dim, 512, 256, 128],
                                       drop_rate=args.drop_rate)

    def forward(self, input_user, input_pos, input_neg):

        user_embs = self.user_encoder(self.user_embedding(input_user))
        pos_embs = self.item_encoder(self.id_embedding(input_pos))
        neg_embs = self.item_encoder(self.id_embedding(input_neg))
        # user_embs = self.user_embedding(input_user)
        # pos_embs = self.id_embedding(input_pos)
        # neg_embs = self.id_embedding(input_neg)
        prediction_i = (user_embs * pos_embs).sum(dim=-1)
        prediction_j = (user_embs * neg_embs).sum(dim=-1)
        return prediction_i, prediction_j
