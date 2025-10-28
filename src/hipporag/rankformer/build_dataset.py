import ast
import random

import igraph as ig
import torch
import pandas as pd
import pickle

from datasets import tqdm

def load_embeddings(entity_path, chunk_path, fact_path):
    entity_df = pd.read_parquet(entity_path)
    chunk_df = pd.read_parquet(chunk_path)
    fact_df = pd.read_parquet(fact_path)
    return entity_df, chunk_df, fact_df


def generate_text_datasets(graph_path, entity_path, chunk_path, fact_path,
                           output_dir="./dataset",
                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Sinh ra 3 file .txt (train, val, test) có format:
    <phrase_id> <passage_id>
    """

    # 1️⃣ Load graph
    with open(graph_path, "rb") as f:
        g: ig.Graph = pickle.load(f)

    # 2️⃣ Load dataset
    entity_df, chunk_df, fact_df = load_embeddings(entity_path, chunk_path, fact_path)

    # 3️⃣ Tạo map node
    node_name_to_idx = {}
    for i, v in enumerate(g.vs):
        name = v["hash_id"] if "hash_id" in v.attributes() else v["name"]
        node_name_to_idx[name] = i

    # 4️⃣ Map entity → chunks
    entity_to_chunks = {}
    for v in g.vs:
        if v["hash_id"].startswith("entity-"):
            e_idx = v.index
            neighbor_idxs = g.neighbors(e_idx, mode="ALL")
            chunk_neighbors = [n for n in neighbor_idxs if g.vs[n]["hash_id"].startswith("chunk-")]
            entity_to_chunks[v["hash_id"]] = chunk_neighbors

    # 5️⃣ Sinh list các cặp (phrase_id, passage_id)
    pairs = []

    for _, row in tqdm(fact_df.iterrows(), total=len(fact_df), desc="Generating pairs"):
        try:
            subj, rel, obj = ast.literal_eval(row["content"])  # parse tuple string
        except Exception:
            continue

        subj, obj = subj.strip().lower(), obj.strip().lower()

        subj_entity = entity_df[entity_df["content"].str.lower() == subj]
        obj_entity = entity_df[entity_df["content"].str.lower() == obj]

        subj_hash = subj_entity["hash_id"].values[0] if not subj_entity.empty else None
        obj_hash = obj_entity["hash_id"].values[0] if not obj_entity.empty else None

        # Nếu có 2 entity tồn tại
        for ent_hash in [subj_hash, obj_hash]:
            if ent_hash and ent_hash in entity_to_chunks:
                for ch_idx in entity_to_chunks[ent_hash]:
                    phrase_id = node_name_to_idx[ent_hash]
                    passage_id = ch_idx
                    pairs.append((phrase_id, passage_id))

    print(f"[INFO] Generated {len(pairs)} phrase–passage pairs")

    # 6️⃣ Shuffle và chia train/val/test
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    # 7️⃣ Ghi ra file .txt
    import os
    os.makedirs(output_dir, exist_ok=True)

    def write_txt(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for pid, cid in data:
                f.write(f"{pid} {cid}\n")

    write_txt(f"{output_dir}/train.txt", train_pairs)
    write_txt(f"{output_dir}/valid.txt", val_pairs)
    write_txt(f"{output_dir}/test.txt", test_pairs)

    print(f"[✅] Saved datasets to {output_dir}")
    print(f" - train.txt: {len(train_pairs)} lines")
    print(f" - valid.txt:   {len(val_pairs)} lines")
    print(f" - test.txt:  {len(test_pairs)} lines")

if __name__ == "__main__":
    root = "../../../outputs/gpt-4o-mini_text-embedding-3-small"
    generate_text_datasets(
        graph_path=f"{root}/graph.pickle",
        entity_path=f"{root}/entity_embeddings/vdb_entity.parquet",
        chunk_path=f"{root}/chunk_embeddings/vdb_chunk.parquet",
        fact_path=f"{root}/fact_embeddings/vdb_fact.parquet",
    )