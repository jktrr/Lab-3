"""
CSCI 114 Lab 3: Graph Neural Networks
Economic trade network analysis with a graph autoencoder.
Group members: Edrian Capistrano, John Kyrk Terrobias, Gabriel Labariento

Environment: Python 3.12.13, PyTorch 2.14.0 (torch, torch.nn, torch.optim only).
Run: python lab3_gnn.py
Every model is seeded with torch.manual_seed(42). Written answers are in lab3_answers.md.
"""


# ============================================================
# Part A
# ============================================================


import torch
import torch.nn as nn

torch.manual_seed(42)

names = ["Aridia",
         "Borland",
         "Cyrenia",
         "Deltora",
         "Estovia",
         "Freland",
         "Galenia",
         "Hesperia"
        ]

A_full = torch.tensor([
    #A  B  C  D  E  F  G  H
    [0, 1, 1, 0, 0, 0, 1, 0], # A
    [1, 0, 0, 0, 0, 1, 1, 0], # B
    [1, 0, 0, 0, 0, 1, 1, 0], # C
    [0, 0, 0, 0, 1, 1, 0, 1], # D
    [0, 0, 0, 1, 0, 0, 0, 1], # E
    [0, 1, 1, 1, 0, 0, 1, 1], # F
    [1, 1, 1, 0, 0, 1, 0, 0], # G
    [0, 0, 0, 1, 1, 1, 0, 0] # H
    ], dtype=torch.float32
)

degree = A_full.sum(dim=1)
degree_ave = torch.mean(degree)
unique_edges = torch.sum(degree)/2

X = torch.tensor([
    [5.8, 3.1, 4.2, 46, 6.2],
    [5.1, 3.8, 5.0, 42, 5.7],
    [6.4, 2.7, 3.8, 51, 7.1],
    [3.2, 6.5, 7.4, 30, 3.1],
    [2.8, 7.2, 8.1, 27, 2.8],
    [4.4, 4.1, 5.6, 39, 4.9],
    [5.6, 3.0, 4.5, 48, 6.5],
    [3.5, 5.8, 6.9, 33, 3.6]
    ], dtype=torch.float32
)

X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X_scaled = (X - X_mean)/X_std

print("=== Part A: Graph and features ===")
# checking the shapes first, X should be 8 countries by 5 features and A should be 8 by 8
print(f"X shape: {X.shape}")
print(f"A_full shape: {A_full.shape}")
# A_full is symmetric if flipping it (A_full.T) gives back the exact same matrix
print(f"Symmetric: {torch.equal(A_full, A_full.T)}")
print(f"Diagonal: {torch.diag(A_full).numpy()}")
print(f"Degree of Country: {degree.numpy()}")
print(f"Mean Degree: {degree_ave.numpy()}")
print(f"Unique Edges: {unique_edges.numpy()}")


# ============================================================
# Part B
# ============================================================


# B1
A_train = torch.tensor([
    #A  B  C  D  E  F  G  H
    [0, 1, 1, 0, 0, 0, 0, 0], # A
    [1, 0, 0, 0, 0, 1, 1, 0], # B
    [1, 0, 0, 0, 0, 1, 1, 0], # C
    [0, 0, 0, 0, 1, 1, 0, 0], # D
    [0, 0, 0, 1, 0, 0, 0, 1], # E
    [0, 1, 1, 1, 0, 0, 1, 1], # F
    [0, 1, 1, 0, 0, 1, 0, 0], # G
    [0, 0, 0, 0, 1, 1, 0, 0] # H
    ], dtype=torch.float32
)

print("\n=== B1: Training graph ===")
print(f"A_train shape: {A_train.shape}")
train_degree = A_train.sum(dim=1)
train_unique_edges = torch.sum(train_degree)/2
print(f"Training Unique Edges: {train_unique_edges.numpy()}")
print(f"Training graph contains 11 unique edges: {train_unique_edges.numpy() == 11.0}")


# B2
def aggregate(A, H):
    I = torch.eye(A.shape[0])
    A_hat = A + I
    degree_hat = A_hat.sum(dim=1)
    degree_hat = degree_hat.clamp(min=1) # handle degrees safely, basically does max(degree_hat, 1)
    D_inverse = torch.diag(1.0 / degree_hat)
    X_aggregated = D_inverse @ A_hat @ H
    assert X_aggregated.shape == H.shape, "X_aggregated's shape == H.shape"
    return X_aggregated


# B3
class GNNEncoder(nn.Module):
    def __init__(self, input_features=5, hidden_size=8, embedding_size=3):
        super().__init__()
        self.linear_one = nn.Linear(input_features, hidden_size)
        self.activation = nn.ReLU()
        self.linear_two = nn.Linear(hidden_size, embedding_size)

    def forward(self, A, X):
        layer_one = self.linear_one(aggregate(A, X))
        X_activated = self.activation(layer_one)
        layer_two = self.linear_two(aggregate(A, X_activated))
        return layer_two

class GraphAutoencoder(nn.Module):
    def __init__(self, input_features_size, hidden_size, embedding_size):
        super().__init__()
        self.encoder = GNNEncoder(input_features_size, hidden_size, embedding_size)

    def forward(self, A, X):
        Z = self.encoder(A, X)
        scores = Z @ Z.T
        A_reconstructed = torch.sigmoid(scores)
        return A_reconstructed, Z


def report(model, A, X):
    model.eval()
    with torch.no_grad():
        probabilities, embeddings = model(A, X)

    print("Reconstructed edge probabilities (diagonal NOT forced to zero):")
    print(probabilities)

    print("\nCountry: Embedding")
    for i, embedding in enumerate(embeddings):
        print(f"{names[i]}: {embedding}")

    return probabilities, embeddings


# B4
torch.manual_seed(42)
INPUT_FEATURES = X_scaled
ENCODER_GRAPH = A_train
BASELINE_RECONSTRUCTION_TARGET = A_train
HIDDEN_DIMENSION = 8
EMBEDDING_DIMENSION = 3
LEARNING_RATE = 0.01
EPOCHS = 2000

model = GraphAutoencoder(INPUT_FEATURES.shape[1], HIDDEN_DIMENSION, EMBEDDING_DIMENSION)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

print("\n=== B4: Training (mean aggregation) ===")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    A_reconstructed, Z = model(ENCODER_GRAPH, INPUT_FEATURES)
    loss = loss_function(A_reconstructed, BASELINE_RECONSTRUCTION_TARGET)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss}")

    if epoch == EPOCHS - 1:
        print(f"Final Loss: {loss}\n")

print(f"Z shape: {Z.shape}")
report(model, ENCODER_GRAPH, INPUT_FEATURES)


# ============================================================
# Part C
# ============================================================


# C1
print("\n=== C1: Hidden-link comparisons ===")
aridia, borland, cyrenia, deltora, estovia, freland, galenia, hesperia = range(8)

with torch.no_grad():
    _, Z = model(ENCODER_GRAPH, INPUT_FEATURES)

    aridia_galenia = torch.sigmoid(Z[aridia] @ Z[galenia])
    aridia_estovia = torch.sigmoid(Z[aridia] @ Z[estovia])
    deltora_hesperia = torch.sigmoid(Z[deltora] @ Z[hesperia])
    deltora_cyrenia = torch.sigmoid(Z[deltora] @ Z[cyrenia])

    print(f"Aridia - Galenia: {aridia_galenia}")
    print(f"Aridia - Estovia: {aridia_estovia}")
    print(f"Hidden real link outranks non-edge: {aridia_galenia > aridia_estovia}\n")

    print(f"Deltora - Hesperia: {deltora_hesperia}")
    print(f"Deltora - Cyrenia: {deltora_cyrenia}")
    print(f"Hidden real link outranks non-edge: {deltora_hesperia > deltora_cyrenia}")


# C2
print("\n=== C2: Candidate ranking ===")

def unique_pairs(A):
    # Undirected edges occur twice in A. Score each pair once, excluding the
    # diagonal: self-loops support aggregation but are not prediction targets.
    return torch.triu_indices(
        A.shape[0], A.shape[1], offset=1,
        device=A.device,
    )

rows, cols = unique_pairs(A_train)
reconstructed_scores = []
with torch.no_grad():
    probabilities, Z = model(ENCODER_GRAPH, INPUT_FEATURES)

    # Because pairs were appended in index order and because Python's sorting is stable,
    # pairs with tied scores stay in increasing pair index order.
    for i, j in zip(rows.tolist(), cols.tolist()):
        if A_train[i, j] == 0:
            reconstructed_scores.append((probabilities[i, j], i, j))
    reconstructed_scores = sorted(reconstructed_scores, key=lambda x:x[0], reverse=True)

for idx, score in enumerate(reconstructed_scores):
    if score[1] == aridia and score[2] == galenia:
        print(f"Aridia - Galenia is Rank {idx + 1}")
    if score[1] == deltora and score[2] == hesperia:
        print(f"Deltora - Hesperia is Rank {idx + 1}\n")

for i in range(5):
    print(f"{names[reconstructed_scores[i][1]]} - {names[reconstructed_scores[i][2]]}: {reconstructed_scores[i][0]}")


# C3
print("\n=== C3: Nearest neighbors and anomaly ranking ===")
country_to_closest_country = {}

with torch.no_grad():
    _, Z = model(ENCODER_GRAPH, INPUT_FEATURES)
    # https://docs.pytorch.org/docs/main/generated/torch.cdist.html
    distances = torch.cdist(Z, Z)

    for i in range(len(distances)):
        closest = float('inf')
        for j in range(len(distances[i])):
            if i == j: continue
            if distances[i, j] < closest:
                closest = distances[i, j]
                country_to_closest_country[i] = (j, closest)

for country in country_to_closest_country:
    print(f"The closest country to {names[country]} is {names[country_to_closest_country[country][0]]}. Distance: {country_to_closest_country[country][1]:.4f}")


with torch.no_grad():
    _, Z = model(ENCODER_GRAPH, INPUT_FEATURES)

    centroid = Z.mean(dim=0)

    centroid_distances = torch.norm(Z - centroid, dim=1)

# rank descending (largest distance = most "anomalous");
# stable sort so ties resolve in country-index order
anomaly_ranking = sorted(
    range(len(names)), key=lambda i: centroid_distances[i].item(), reverse=True
)

print(f"\nEmbedding centroid: {centroid}\n")
print("Rank  Country     Distance to centroid")
for rank, i in enumerate(anomaly_ranking, start=1):
    print(f"{rank:<5} {names[i]:<11} {centroid_distances[i]:.4f}")


# ============================================================
# Part D
# ============================================================


# D1
print("\n=== D1: Borland inflation shock ===")

X_changed = X.clone()
X_changed[borland, 1] = 12.0

X_changed_scaled = (X_changed - X_mean)/X_std

with torch.no_grad():
    Z_before = model.encoder(A_train, X_scaled)
    Z_after  = model.encoder(A_train, X_changed_scaled)

    print("Country: ||Z_after - Z_before||_2")
    for c in [borland, aridia, freland, galenia, estovia]:
        change = torch.norm(Z_after[c] - Z_before[c])
        print(f"{names[c]:>7}: {change:.4f}")


# D2
print("\n=== D2: Feature-only baseline ===")
class BaselineEncoder(nn.Module):
    def __init__(self, input_features=5, hidden_size=8, embedding_size=3):
        super().__init__()
        self.linear_one = nn.Linear(input_features, hidden_size)
        self.activation = nn.ReLU()
        self.linear_two = nn.Linear(hidden_size, embedding_size)

    def forward(self, X):
        return self.linear_two(self.activation(self.linear_one(X)))

torch.manual_seed(42)
baseline = BaselineEncoder(INPUT_FEATURES.shape[1], HIDDEN_DIMENSION, EMBEDDING_DIMENSION)

with torch.no_grad():
    Z_baseline = baseline(X_scaled)
    print(f"Baseline embedding shape: {Z_baseline.shape}")


# ============================================================
# Part E
# ============================================================


A_train = torch.tensor([
    #A  B  C  D  E  F  G  H
    [0, 1, 1, 0, 0, 0, 0, 0], # A
    [1, 0, 0, 0, 0, 1, 1, 0], # B
    [1, 0, 0, 0, 0, 1, 1, 0], # C
    [0, 0, 0, 0, 1, 1, 0, 0], # D
    [0, 0, 0, 1, 0, 0, 0, 1], # E
    [0, 1, 1, 1, 0, 0, 1, 1], # F
    [0, 1, 1, 0, 0, 1, 0, 0], # G
    [0, 0, 0, 0, 1, 1, 0, 0] # H
    ], dtype=torch.float32
)

def aggregate_symmetric(A, H):
    I = torch.eye(A.shape[0])
    A_hat = A + I
    degree_hat = A_hat.sum(dim=1)
    degree_hat = degree_hat.clamp(min=1) # handle degrees safely, basically does max(degree_hat, 1)
    d_inv_sqrt = torch.pow(degree_hat, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    X_aggregated = D_inv_sqrt @ A_hat @ D_inv_sqrt @ H
    assert X_aggregated.shape == H.shape, "X_aggregated's shape == H.shape"

    return X_aggregated

class SymmetricGNNEncoder(nn.Module):
    def __init__(self, input_features=5, hidden_size=8, embedding_size=3):
        super().__init__()
        self.linear_one = nn.Linear(input_features, hidden_size)
        self.activation = nn.ReLU()
        self.linear_two = nn.Linear(hidden_size, embedding_size)

    def forward(self, A, X):
        layer_one = self.linear_one(aggregate_symmetric(A, X))
        X_activated = self.activation(layer_one)
        layer_two = self.linear_two(aggregate_symmetric(A, X_activated))
        return layer_two

class SymmetricGraphAutoencoder(nn.Module):
    def __init__(self, input_features_size, hidden_size, embedding_size):
        super().__init__()
        self.encoder = SymmetricGNNEncoder(input_features_size, hidden_size, embedding_size)

    def forward(self, A, X):
        Z = self.encoder(A, X)
        scores = Z @ Z.T
        A_reconstructed = torch.sigmoid(scores)
        return A_reconstructed, Z

torch.manual_seed(42)
INPUT_FEATURES = X_scaled
ENCODER_GRAPH = A_train
BASELINE_RECONSTRUCTION_TARGET = A_train
HIDDEN_DIMENSION = 8
EMBEDDING_DIMENSION = 3
LEARNING_RATE = 0.01
EPOCHS = 2000

model_new = SymmetricGraphAutoencoder(INPUT_FEATURES.shape[1], HIDDEN_DIMENSION, EMBEDDING_DIMENSION)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(
    model_new.parameters(), lr=LEARNING_RATE
)

print("\n=== Part E: Training (symmetric normalization) ===")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    A_reconstructed, Z = model_new(ENCODER_GRAPH, INPUT_FEATURES)
    loss = loss_function(A_reconstructed, BASELINE_RECONSTRUCTION_TARGET)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss}")

    if epoch == EPOCHS - 1:
        print(f"Final Loss: {loss}\n")

report(model_new, ENCODER_GRAPH, INPUT_FEATURES)


# E comparison
# same checks as C1-C3, but this time on model_new so we can compare it with the original model
print("\n=== Part E: Comparison with the original model ===")
with torch.no_grad():
    probabilities_new, Z_new = model_new(ENCODER_GRAPH, INPUT_FEATURES)

print(f"Aridia - Galenia: {probabilities_new[aridia, galenia]}")
print(f"Aridia - Estovia: {probabilities_new[aridia, estovia]}")
print(f"Deltora - Hesperia: {probabilities_new[deltora, hesperia]}")
print(f"Deltora - Cyrenia: {probabilities_new[deltora, cyrenia]}\n")

rows, cols = unique_pairs(A_train)
reconstructed_scores_new = []
for i, j in zip(rows.tolist(), cols.tolist()):
    if A_train[i, j] == 0:
        reconstructed_scores_new.append((probabilities_new[i, j], i, j))
reconstructed_scores_new = sorted(reconstructed_scores_new, key=lambda x:x[0], reverse=True)

for idx, score in enumerate(reconstructed_scores_new):
    if score[1] == aridia and score[2] == galenia:
        print(f"Aridia - Galenia is Rank {idx + 1}")
    if score[1] == deltora and score[2] == hesperia:
        print(f"Deltora - Hesperia is Rank {idx + 1}")
print()

distances_new = torch.cdist(Z_new, Z_new)
for i in range(len(names)):
    closest = float('inf')
    for j in range(len(names)):
        if i == j: continue
        if distances_new[i, j] < closest:
            closest = distances_new[i, j]
            closest_country = j
    print(f"The closest country to {names[i]} is {names[closest_country]}. Distance: {closest:.4f}")

centroid_new = Z_new.mean(dim=0)
centroid_distances_new = torch.norm(Z_new - centroid_new, dim=1)
anomaly_ranking_new = sorted(
    range(len(names)), key=lambda i: centroid_distances_new[i].item(), reverse=True
)

print("\nRank  Country     Distance to centroid")
for rank, i in enumerate(anomaly_ranking_new, start=1):
    print(f"{rank:<5} {names[i]:<11} {centroid_distances_new[i]:.4f}")
