import torch
import torch.nn as nn


class DLRM(nn.Module):
    def __init__(
        self, bottom_mlp_sizes, top_mlp_sizes, embedding_sizes, interaction_op="dot"
    ):
        super(DLRM, self).__init__()
        self.interaction_op = interaction_op

        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, embedding_dim)
                for num_embeddings, embedding_dim in embedding_sizes
            ]
        )

        # Bottom MLP
        layers = []
        for i in range(len(bottom_mlp_sizes) - 1):
            layers.append(nn.Linear(bottom_mlp_sizes[i], bottom_mlp_sizes[i + 1]))
            if i < len(bottom_mlp_sizes) - 2:
                layers.append(nn.ReLU())
        self.bottom_mlp = nn.Sequential(*layers)

        # Top MLP
        if interaction_op == "dot":
            interaction_output_size = (
                len(embedding_sizes) * (len(embedding_sizes) + 1) // 2
            )  # embeddings + dense dot products
        elif interaction_op == "cat":
            interaction_output_size = (
                len(embedding_sizes) * bottom_mlp_sizes[-1]
            )  # embeddings concatenation
        else:
            raise ValueError("Unsupported interaction operation")

        layers = []
        top_input_size = (
            interaction_output_size + bottom_mlp_sizes[-1]
        )  # interaction + dense feature
        top_mlp_sizes = [top_input_size] + top_mlp_sizes
        for i in range(len(top_mlp_sizes) - 1):
            layers.append(nn.Linear(top_mlp_sizes[i], top_mlp_sizes[i + 1]))
            if i < len(top_mlp_sizes) - 2:
                layers.append(nn.ReLU())
        self.top_mlp = nn.Sequential(*layers)

    def interact_features(self, dense, embeddings):
        # Combine dense feature with embeddings
        features = [dense] + embeddings
        if self.interaction_op == "dot":
            # Pairwise dot product
            interactions = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    interactions.append(
                        (features[i] * features[j]).sum(dim=1, keepdim=True)
                    )
            return torch.cat([dense] + interactions, dim=1)
        elif self.interaction_op == "cat":
            return torch.cat(features, dim=1)
        else:
            raise ValueError("Unknown interaction op")

    def forward(self, dense_x, sparse_indices):
        # Dense features
        dense_out = self.bottom_mlp(dense_x)
        # Sparse features
        embedded = [emb(sparse_indices[:, i]) for i, emb in enumerate(self.embeddings)]
        # Interaction
        x = self.interact_features(dense_out, embedded)
        # Top MLP
        out = self.top_mlp(x)
        return torch.sigmoid(out)


if __name__ == "__main__":
    # Define model
    num_dense_features = 13
    num_sparse_features = 26
    embedding_dim = 16
    model = DLRM(
        bottom_mlp_sizes=[
            num_dense_features,
            512,
            256,
            64,
            embedding_dim,
        ],  # dense features
        top_mlp_sizes=[512, 256, 1],
        embedding_sizes=[(1000, embedding_dim)]
        * num_sparse_features,  # categorical features, set num_embeddings=1000
        interaction_op="dot",
    )

    # Dummy input
    dense_input = torch.rand(
        128, num_dense_features
    )  # batch_size=128, dense features=13
    sparse_input = torch.randint(
        low=0, high=1000, size=(128, num_sparse_features)
    )  # batch_size=128, 26 categorical features

    output = model(dense_input, sparse_input)
    print("Model output:", output)
