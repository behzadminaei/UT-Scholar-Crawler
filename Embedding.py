from Tools import *

# Import centralized Persian text utilities for consistent text rendering
from utils.persian_text import fix_persian_text, configure_persian_matplotlib

# Configure matplotlib for Persian text support
configure_persian_matplotlib()

Nodes = Load_JSON('Outputs/Aggregated_Nodes.json')
Edges = Load_JSON('Outputs/Aggregated_Edges.json')

Node_ID_2_Embedding = {}

Node_IDs = list(map(int, Nodes.keys()))

for Node_ID in Node_IDs:
    Embedding = np.random.uniform(low=-1, high=+1, size=(EMBEDDING_SIZE))
    Embedding = Embedding / np.linalg.norm(x=Embedding, ord=2)
    Node_ID_2_Embedding[Node_ID] = Embedding

Node_ID_2_Embedding_Temp = {Node_ID: np.zeros(shape=(EMBEDDING_SIZE, ), dtype=np.float32) for Node_ID in Node_IDs}

Losses = []
Iterations = np.arange(start=1, stop=ITERATION_COUNT + 1, step=1, dtype=np.int32)

for Iteration in Iterations:
    for Node_ID_A, Node_ID_Bs in Edges.items():
        Node_ID_A = int(Node_ID_A)
        for Node_ID_B in Node_ID_Bs:
            Node_ID_2_Embedding_Temp[Node_ID_A] += Node_ID_2_Embedding[Node_ID_B]
            Node_ID_2_Embedding_Temp[Node_ID_B] += Node_ID_2_Embedding[Node_ID_A]
    Node_ID_2_Embedding_New = {k: v / (np.linalg.norm(x=v, ord=2) + EPSILON) for k, v in Node_ID_2_Embedding_Temp.items()}
    Loss = []
    for k, v in Node_ID_2_Embedding.items():
        v_new = Node_ID_2_Embedding_New[k]
        Loss.append(np.mean(np.abs(np.subtract(v, v_new))))
    Loss = np.mean(a=Loss)
    Losses.append(Loss)
    print(f'Iteration {Iteration} -> Loss {Loss:.6f}')
    Node_ID_2_Embedding = co.deepcopy(Node_ID_2_Embedding_New)

# Save embeddings for Hub Finder analysis
print('\nSaving embeddings for Hub Finder...')
embeddings_to_save = {
    int(node_id): embedding.tolist()
    for node_id, embedding in Node_ID_2_Embedding.items()
}

with open('Outputs/Node_Embeddings.json', 'w') as f:
    js.dump(embeddings_to_save, f)

print(f'Saved {len(embeddings_to_save)} embeddings (dim={EMBEDDING_SIZE}) to Outputs/Node_Embeddings.json')

plt.plot(Iterations, Losses, ls='-', lw=1.2, c='crimson')
plt.title(label='Embedding Loss Over Iterations')
plt.xlabel(xlabel='Iteration')
plt.ylabel(ylabel='Loss')
plt.yscale(value='log')
plt.show()

Embeddings = np.array(object=list(Node_ID_2_Embedding.values()))

PCA = man.TSNE(n_components=2,
               perplexity=30,
               learning_rate='auto',
               max_iter=1000,
               n_iter_without_progress=300,
               init='pca',
               random_state=SEED)
Embeddings_2D = PCA.fit_transform(X=Embeddings)

Edge_Lengths_2D = []
for Node_ID_A, Node_ID_Bs in Edges.items():
    Node_ID_A = int(Node_ID_A)
    Node_Index_A = Node_IDs.index(Node_ID_A)
    Embedding_2D_A = Embeddings_2D[Node_Index_A, :]
    for Node_ID_B in Node_ID_Bs:
        Node_Index_B = Node_IDs.index(Node_ID_B)
        Embedding_2D_B = Embeddings_2D[Node_Index_B, :]
        Length = np.linalg.norm(x=np.subtract(Embedding_2D_A, Embedding_2D_B), ord=2)
        Edge_Lengths_2D.append(Length)

Length_2D_Threshold = np.percentile(a=Edge_Lengths_2D, q=90)

for Node_ID_A, Node_ID_Bs in Edges.items():
    Node_ID_A = int(Node_ID_A)
    Node_Index_A = Node_IDs.index(Node_ID_A)
    Embedding_2D_A = Embeddings_2D[Node_Index_A, :]
    for Node_ID_B in Node_ID_Bs:
        Node_Index_B = Node_IDs.index(Node_ID_B)
        Embedding_2D_B = Embeddings_2D[Node_Index_B, :]
        Length = np.linalg.norm(x=np.subtract(Embedding_2D_A, Embedding_2D_B), ord=2)
        if Length <= Length_2D_Threshold:
            Xs = [Embedding_2D_A[0], Embedding_2D_B[0]]
            Ys = [Embedding_2D_A[1], Embedding_2D_B[1]]
            plt.plot(Xs, Ys, ls='-', lw=0.8, c='crimson')
plt.scatter(x=Embeddings_2D[:, 0], y=Embeddings_2D[:, 1], s=10, c='teal')
plt.title(label='Teachers Embedding 2D Representation')
plt.xlabel(xlabel='X1')
plt.ylabel(ylabel='X2')
plt.show()

print('END')
print('\n' + '='*70)
print('Embeddings saved! You can now run:')
print('  python integrate_with_existing.py')
print('to identify research hubs and key individuals.')
print('='*70)