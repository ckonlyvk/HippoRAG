import os
from random import random

from .parse import parse_args, args
from .dataloader import MyDataset
from .model import Model
import torch
import numpy as np

def recommend_for_users(model, user_ids, weights=None, top_k=10, candidate_k=100):
    """
    Hybrid retrieval + rerank phù hợp với Model / Rankformer trong project của bạn.

    Args:
        model: instance của class Model (không phải Rankformer trực tiếp).
               Phải có: model._users, model._items, model.Rankformer, model.dataset
        user_ids: 1D list/torch tensor các user id (những entity nodes)
        weights: list/1D-tensor trọng số tương ứng với user_ids (None -> đều nhau)
        top_k: số item cuối cùng trả về
        candidate_k: số candidate dùng để rerank

    Returns:
        final_items: list item IDs (int)
        final_scores: list float scores (tương ứng)
    """
    model.eval()

    # --- đảm bảo embedding đã compute ---
    if model._users is None or model._items is None:
        if hasattr(model, "computer"):
            model.computer()
        else:
            raise ValueError("Model chưa có embedding tĩnh (_users/_items) và không có method computer().")

    device = next(model.parameters()).device
    # cast user_ids -> tensor trên device
    if not isinstance(user_ids, torch.Tensor):
        user_ids = torch.tensor(user_ids, device=device)
    else:
        user_ids = user_ids.to(device)

    # --- lấy embedding người dùng (tĩnh retrieval embedding) ---
    user_embs = model._users[user_ids]              # [num_users, dim]

    # --- weighted pooling để tạo group embedding ---
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=device)
        else:
            weights = weights.to(device)
        weights = weights.squeeze()
        if weights.dim() != 1 or weights.shape[0] != user_embs.shape[0]:
            raise ValueError("weights phải là 1D và cùng độ dài với user_ids")
        w = weights.unsqueeze(1).to(dtype=user_embs.dtype)
        w = w / (w.sum() + 1e-12)
        group_emb = (user_embs * w).sum(dim=0)     # [dim]
    else:
        group_emb = user_embs.mean(dim=0)           # [dim]

    # --- retrieval bằng embedding tĩnh (nhanh) ---
    items_emb = model._items                       # [num_items, dim]
    # Nếu embeddings chưa ở cùng dtype/device, đảm bảo
    items_emb = items_emb.to(device=device, dtype=group_emb.dtype)
    retrieval_scores = torch.matmul(items_emb, group_emb)  # [num_items]
    candidate_k = min(candidate_k, items_emb.shape[0])
    top_scores, top_items = torch.topk(retrieval_scores, k=candidate_k)

    # --- rerank: tạo all_emb và chạy Rankformer để get refined embeddings ---
    # all_emb giống cách model.computer() tạo: concat users + items
    all_emb = torch.cat([model._users.to(device=device, dtype=items_emb.dtype),
                         model._items.to(device=device, dtype=items_emb.dtype)], dim=0)

    # dùng same (u, i) edges mà model dùng (training edges) để Rankformer hoạt động
    u_edges = model.dataset.train_user.to(device)
    i_edges = model.dataset.train_item.to(device)

    # compute refined embedding bằng Rankformer
    # Rankformer.forward expects (x, u, i) per your code
    with torch.no_grad():
        rec_emb = model.Rankformer(all_emb, u_edges, i_edges)

    # combine like in computer(): all_emb = (1 - tau) * all_emb + tau * rec_emb
    tau = getattr(args, "rankformer_tau", 1.0) if 'args' in globals() else 1.0
    refined_all = (1.0 - tau) * all_emb + tau * rec_emb
    # split refined item embeddings
    n_users = model.dataset.num_users
    refined_items = refined_all[n_users:]  # [num_items, dim]

    # --- compute rerank scores only on candidates ---
    cand_items_emb = refined_items[top_items]   # [candidate_k, dim]
    # compute dot product between group_emb and each candidate item
    group_emb_for_score = group_emb.to(dtype=cand_items_emb.dtype)
    rerank_scores = torch.matmul(cand_items_emb, group_emb_for_score)  # [candidate_k]

    # --- final top-k from reranked candidates ---
    k = min(top_k, rerank_scores.shape[0])
    final_scores, idx_in_candidates = torch.topk(rerank_scores, k=k)
    final_items = top_items[idx_in_candidates]

    return final_items.cpu().tolist(), final_scores.cpu().tolist()

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