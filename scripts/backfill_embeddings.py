"""Backfill embeddings, PCA, and cluster IDs via Ollama bge-m3.

Usage:
    py -3.11 scripts/backfill_embeddings.py <db_path>
"""
import sys, json, sqlite3, urllib.request
import numpy as np
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"

def embed_batch(texts):
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    return np.array(data["embeddings"], dtype=np.float32)

def main():
    if len(sys.argv) < 2:
        print("Usage: py -3.11 scripts/backfill_embeddings.py <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    if not Path(db_path).exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, code FROM programs WHERE code IS NOT NULL").fetchall()
    print(f"[1/4] {len(rows)} programs in {db_path}")

    if not rows:
        conn.close()
        return

    ids = [r[0] for r in rows]
    codes = [r[1] for r in rows]

    # Embed via Ollama bge-m3
    print(f"[2/4] Embedding with {MODEL}...")
    BATCH = 8
    all_emb = []
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i+BATCH]
        emb = embed_batch(batch)
        all_emb.append(emb)
        print(f"  {min(i+BATCH, len(codes))}/{len(codes)}")
    embeddings = np.vstack(all_emb)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    print(f"  Shape: {embeddings.shape}")

    # PCA + KMeans
    print("[3/4] PCA 2D/3D + KMeans...")
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    n = embeddings.shape[0]
    if n >= 3:
        pca = PCA(n_components=min(3, n))
        c3d = pca.fit_transform(embeddings)
        c2d = c3d[:, :2]
        print(f"  Variance explained: {[f'{v:.2%}' for v in pca.explained_variance_ratio_]}")
    else:
        c2d = np.zeros((n, 2))
        c3d = np.zeros((n, 3))

    n_clusters = min(max(2, n // 3), 8)
    if n >= n_clusters:
        clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(embeddings).tolist()
    else:
        clusters = list(range(n))

    # Write
    print(f"[4/4] Writing ({n_clusters} clusters)...")
    for i, pid in enumerate(ids):
        conn.execute(
            "UPDATE programs SET embedding=?, embedding_pca_2d=?, embedding_pca_3d=?, embedding_cluster_id=? WHERE id=?",
            (json.dumps(embeddings[i].tolist()), json.dumps(c2d[i].tolist()), json.dumps(c3d[i].tolist()), clusters[i], pid)
        )
    conn.commit()
    conn.close()
    print(f"Done! {n} programs, dim={embeddings.shape[1]}, clusters={n_clusters}")

if __name__ == "__main__":
    main()
