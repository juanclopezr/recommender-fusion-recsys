from torch import nn
import torch
from architectures.mlp import MultiLayerPerceptron
from architectures.generalized_matrix_factorization import GeneralizedMatrixFactorization


class Graph(nn.Module):
    def __init__(self, args, num_users, num_items, use_item_embedding=True):
        super(Graph, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mlp =  int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.mlp = MultiLayerPerceptron(num_users, num_items, self.factor_num_mlp, self.layers)
        
        if use_item_embedding:
            self.affine_output = nn.Linear(in_features=args.layers[-1] + 256, out_features=1)
        else:
            self.affine_output = nn.Linear(in_features=args.layers[-1] + 128, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        self.mlp.init_weight()

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices, graph_embeddings):
        mlp_vector = self.mlp(user_indices, item_indices)

        graph_vector = graph_embeddings

        vector = torch.cat([mlp_vector, graph_vector], dim=1)
        
        #print(mlp_vector.shape, mf_vector.shape, bert_vector.shape, graph_vector.shape)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()