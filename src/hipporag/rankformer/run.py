import os
from random import random

from .parse import parse_args, args
from .dataloader import MyDataset
from .model import Model
import torch
import numpy as np

def recommend_for_users(model, user_ids, weights=None, top_k=10):
    """
    Recommend top-K items cho nhóm user có trọng số.

    Args:
        model: Model đã train (đã có _users và _items)
        user_ids: list hoặc tensor các user ID
        weights: list hoặc tensor trọng số (nếu None → đều nhau)
        top_k: số lượng item cần gợi ý

    Returns:
        top_items: list chứa ID các item được recommend
        scores: list chứa điểm tương ứng
    """
    model.eval()

    # Đảm bảo đã tính embedding
    if model._users is None or model._items is None:
        model.computer()

    device = next(model.parameters()).device
    user_ids = torch.tensor(user_ids, device=device)

    # Lấy embedding các user trong nhóm
    user_embs = model._users[user_ids]  # shape: [num_users, dim]

    # Nếu có trọng số, chuẩn hóa rồi gộp
    if weights is not None:
        weights = torch.tensor(weights, device=device).unsqueeze(1)
        weights = weights / weights.sum()  # chuẩn hóa
        group_emb = (user_embs * weights.float()).sum(dim=0)  # weighted sum
    else:
        group_emb = user_embs.mean(dim=0)  # trung bình đều

    # Tính score cho tất cả item
    scores = torch.matmul(model._items, group_emb)

    # Lấy top-k item
    top_scores, top_items = torch.topk(scores, k=top_k)

    return top_items.tolist(), top_scores.tolist()

def build_rank_former() -> Model:
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

    dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)

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
    torch.save(model.state_dict(), "saved_model.pt")
    return model

if __name__ == "__main__":
    build_rank_former()