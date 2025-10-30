import os
from random import random

from .parse import parse_args, args
from .dataloader import MyDataset
from .model import Model
import torch
import numpy as np

def recommend_for_users(model, user_ids, weights=None, top_k=10, candidate_k=100):
    """
    Recommend top-K item cho nhóm user có trọng số bằng RankFormer (retrieval + rerank).

    Args:
        model: RankFormer đã train (phải có _users, _items, rank(), và encode_users())
        user_ids: list các user ID
        weights: list trọng số (nếu None → đều nhau)
        top_k: số lượng item cuối cùng trả về
        candidate_k: số lượng item dùng để rerank (retrieval ban đầu)

    Returns:
        final_items: list ID item được recommend
        final_scores: list score tương ứng (sau rerank)
    """

    model.eval()

    # --- 1️⃣ Đảm bảo đã có embedding retrieval ---
    if model._users is None or model._items is None:
        if hasattr(model, "compute"):
            model.compute()  # hoặc model.compute_embeddings()
        else:
            raise ValueError("Model chưa có embedding tĩnh (_users, _items).")

    device = next(model.parameters()).device
    user_ids = torch.tensor(user_ids, device=device)

    # --- 2️⃣ Lấy embedding người dùng ---
    user_embs = model._users[user_ids]  # [num_users, dim]

    # --- 3️⃣ Gộp embedding theo trọng số (group embedding) ---
    if weights is not None:
        weights = torch.tensor(weights, device=device).unsqueeze(1)
        assert weights.shape[0] == user_embs.shape[0], "weights phải khớp với số user"
        weights = weights / weights.sum()
        weights = weights.to(dtype=user_embs.dtype)
        group_emb = (user_embs * weights).sum(dim=0)
    else:
        group_emb = user_embs.mean(dim=0)

    # --- 4️⃣ Retrieval bằng embedding tĩnh ---
    # Có thể normalize nếu RankFormer train bằng cosine loss
    # group_emb = F.normalize(group_emb, dim=0)
    # items_emb = F.normalize(model._items, dim=1)
    items_emb = model._items

    retrieval_scores = torch.matmul(items_emb, group_emb)  # [num_items]
    top_scores, top_items = torch.topk(retrieval_scores, k=candidate_k)

    # --- 5️⃣ Rerank bằng transformer RankFormer ---
    # Encode lại group embedding qua transformer encoder (nếu có)
    if hasattr(model, "encode_users"):
        group_emb = model.encode_users(user_ids, weights=weights)

    # Gọi hàm rank (chuẩn RankFormer) để rerank top-K’ item
    if hasattr(model, "rank"):
        reranked_scores = model.rank(group_emb, top_items)
    elif hasattr(model, "forward"):
        reranked_scores = model.forward(group_emb, top_items)
    else:
        raise ValueError("Model không có hàm rank() hoặc forward() để rerank.")

    # --- 6️⃣ Chọn top-K cuối cùng ---
    reranked_scores = reranked_scores.squeeze()
    final_scores, final_idx = torch.topk(reranked_scores, k=top_k)
    final_items = top_items[final_idx]

    return final_items.tolist(), final_scores.tolist()

def build_rank_former(train_file, valid_file, test_file, model_dir) -> Model:
    best_valid_ndcg, best_epoch = 0., 0
    test_pre, test_recall, test_ndcg = torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks))

    def print_test_result():
        nonlocal best_epoch, test_pre, test_recall, test_ndcg
        print(f'===== Test Result(at {best_epoch:d} epoch) =====')
        for i, k in enumerate(args.topks):
            print(f'ndcg@{k:d} = {test_ndcg[i]:f}, recall@{k:d} = {test_recall[i]:f}, pre@{k:d} = {test_pre[i]:f}')

    def train():
        train_loss = model.train_func()
        if epoch % args.show_loss_interval == 0:
            print(f'epoch {epoch:d}, train_loss = {train_loss:f}.')

    def valid(epoch):
        nonlocal best_valid_ndcg, best_epoch, test_pre, test_recall, test_ndcg
        valid_pre, valid_recall, valid_ndcg = model.valid_func()
        for i, k in enumerate(args.topks):
            print(
                f'[{epoch:d}/{args.max_epochs:d}] Valid Result: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}, pre@{k:d} = {valid_pre[i]:f}.')
        if valid_ndcg[-1] > best_valid_ndcg:
            best_valid_ndcg, best_epoch = valid_ndcg[-1], epoch
            test_pre, test_recall, test_ndcg = model.test_func()
            print_test_result()
            if args.save_emb:
                model.save_emb()
            return True
        return False

    dataset = MyDataset(train_file, valid_file, test_file, args.device)

    model = Model(dataset).to(args.device)
    if args.load_emb:
        model.load_emb()

    valid(epoch=0)
    for epoch in range(1, args.max_epochs+1):
        train()
        if epoch % args.valid_interval == 0:
            if not valid(epoch) and epoch-best_epoch >= args.stopping_step*args.valid_interval:
                break
    print('---------------------------')
    print('done.')
    print_test_result()
    torch.save(model, model_dir)
    return model

# if __name__ == "__main__":
#     build_rank_former()