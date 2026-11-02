import torch
from torch import nn


names = ["Alice", "Bob", "Carol", "David", "Eve"]
# Original graph from example_1 through example_4.
A = torch.tensor([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=torch.float32)

# A second input graph: add Alice--Eve in BOTH directions.
# This connection was a non-edge during training.
A_changed = torch.tensor([
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0],
], dtype=torch.float32)

# [age, posts/week, purchases/month]
X = torch.tensor([
    [25, 4, 10],
    [30, 2, 7],
    [22, 6, 12],
    [35, 1, 5],
    [28, 3, 8],
], dtype=torch.float32)


class GNNEncoder(nn.Module):
    def __init__(self, input_features=3, hidden_size=32, embedding_size=8):
        super().__init__()
        # Retain each node's own features alongside its neighborhood mean.
        # Averaging alone can erase distinctions between different nodes.
        # A nonlinear encoder has more capacity than example_4's single linear
        # layer and two-dimensional embeddings.
        self.network = nn.Sequential(
            nn.Linear(2 * input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, adjacency, features):
        with_self_loops = adjacency + torch.eye(
            adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
        )
        degree = with_self_loops.sum(dim=1, keepdim=True)
        neighborhood_mean = (with_self_loops @ features) / degree
        return self.network(torch.cat([features, neighborhood_mean], dim=1))


class GraphAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GNNEncoder()

    def forward(self, adjacency, features):
        embeddings = self.encoder(adjacency, features)
        # Return logits for numerically stable BCEWithLogitsLoss.
        return embeddings @ embeddings.T, embeddings


def unique_pairs(adjacency):
    # Undirected edges occur twice in A. Score each pair once, excluding the
    # diagonal: self-loops support aggregation but are not prediction targets.
    return torch.triu_indices(
        adjacency.shape[0], adjacency.shape[1], offset=1,
        device=adjacency.device,
    )


def report(model, adjacency, features, title):
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(adjacency, features)
        probabilities = torch.sigmoid(logits)
        rows, cols = unique_pairs(adjacency)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[rows, cols], adjacency[rows, cols]
        ).item()
        accuracy = (
            (probabilities[rows, cols] >= 0.5) == adjacency[rows, cols].bool()
        ).float().mean().item()
        # Z_i dot Z_i is nonnegative, so its sigmoid cannot model a zero
        # diagonal. Set the diagonal to zero for displaying this simple graph.
        probabilities.fill_diagonal_(0)

    print(f"\n{title}")
    print("Supplied adjacency matrix:")
    print(adjacency)
    print("Reconstructed edge probabilities (diagonal forced to zero):")
    print(probabilities)
    print("Reconstructed adjacency at threshold 0.5:")
    print((probabilities >= 0.5).int())
    print(f"Pairwise BCE: {loss:.4f}; pairwise accuracy: {accuracy:.0%}")
    print("Edge                 Target    Score")
    for i, j in zip(rows.tolist(), cols.tolist()):
        print(f"{names[i] + '--' + names[j]:20s}"
              f" {int(adjacency[i, j]):3d}      {probabilities[i, j]:.4f}")
    return probabilities, embeddings


torch.manual_seed(42)
# One CPU thread avoids parallel overhead for these tiny matrices.
torch.set_num_threads(1)
torch.set_printoptions(precision=4, sci_mode=False)

# Fit scaling on training features and reuse it for the changed graph.
features = (X - X.mean(dim=0)) / X.std(dim=0, unbiased=False).clamp_min(1e-6)
model = GraphAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.BCEWithLogitsLoss()
rows, cols = unique_pairs(A)

print("Training ONLY on the original adjacency matrix...")
model.train()
for epoch in range(1, 2001):
    optimizer.zero_grad()
    logits, _ = model(A, features)
    loss = loss_function(logits[rows, cols], A[rows, cols])
    loss.backward()
    optimizer.step()
    if epoch == 1 or epoch % 400 == 0:
        print(f"Epoch {epoch:4d}: loss = {loss.item():.6f}")

original_scores, _ = report(model, A, features, "EXAMPLE 1: Original graph")
changed_scores, _ = report(
    model, A_changed, features,
    "EXAMPLE 2: Added Alice--Eve connection (no retraining)",
)

alice, bob, eve = 0, 1, 4
print("\nConnection comparison:")
print(f"Familiar Alice--Bob, original input: {original_scores[alice, bob]:.4f}")
print(f"Familiar Alice--Bob, changed input:  {changed_scores[alice, bob]:.4f}")
print(f"Absent Alice--Eve, original input:   {original_scores[alice, eve]:.4f}")
print(f"Added Alice--Eve, changed input:     {changed_scores[alice, eve]:.4f}")
print(
    "\nThe changed matrix is actually passed into the encoder, so it changes"
    "\nthe neighborhood averages and embeddings. The weights remain fixed."
    "\nA low score on the added edge means the learned reconstruction does"
    "\nnot support it strongly, despite its presence in the supplied matrix."
    "\nThis is a toy reconstruction demonstration, not a held-out link test:"
    "\nAlice--Eve was explicitly trained as a non-edge. Other graph changes"
    "\nare not guaranteed to receive low scores."
)
