import os
from random import random

from .parse import parse_args, args
from .dataloader import MyDataset
from .model import Model
import torch
import numpy as np

def recommend_for_users(model, user_ids, weights=None, top_k=50):
    """
    Sinh gợi ý item cho 1 nhóm user bằng cách trung bình score thay vì trung bình embedding.
    - model: instance của class Model
    - user_ids: tensor các user id
    - weights: trọng số (tensor có cùng chiều với user_ids) hoặc None
    - top_k: số lượng item muốn lấy ra
    """
    model.eval()
    with torch.no_grad():
        # Lấy embedding user & item từ model đã huấn luyện
        user_embs = model._users[user_ids]    # (num_users, dim)
        item_embs = model._items              # (num_items, dim)

        # Tính score từng user-item
        scores = item_embs @ user_embs.T      # (num_items, num_users)

        # Nếu có trọng số, áp dụng trọng số theo user
        if weights is not None:
            weights = weights / weights.sum()  # chuẩn hóa
            scores = scores * weights          # broadcasting tự động (num_items, num_users)

        # Trung bình score của các user
        mean_scores = scores.mean(dim=1)       # (num_items,)

        # Lấy top_k item tốt nhất
        top_scores, top_items = torch.topk(mean_scores, k=top_k)
        return top_items, top_scores

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