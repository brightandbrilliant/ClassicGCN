from BlogCatalog_parse import read_data
from BlogCatalog_build import preprocess_social_graph
import random
import os
import torch
from collections import defaultdict, Counter

def extract_global_group_map(user_dict):
    all_groups = set()
    for info in user_dict.values():
        all_groups.update(info.get('groups', []))
    all_groups = sorted(list(all_groups))
    return {gid: idx for idx, gid in enumerate(all_groups)}

def split_user_dict_with_domain_shift(user_dict, n_clients=3, dominant_ratio=0.75, test_ratio=0.1, seed=42, groups_per_client=20):
    random.seed(seed)

    # 1. 建立 group -> users 映射
    group_to_users = defaultdict(set)
    for uid, info in user_dict.items():
        for g in info['groups']:
            group_to_users[g].add(uid)

    all_groups = list(group_to_users.keys())
    random.shuffle(all_groups)

    # 2. 将 group 均匀分配给每个 client
    group_clusters = [all_groups[i::n_clients] for i in range(n_clients)]
    # 每个 cluster 中最多取 groups_per_client 个 group 作为 dominant
    client_dominant_groups = [cluster[:groups_per_client] for cluster in group_clusters]

    clients_users = [set() for _ in range(n_clients)]
    all_users = set(user_dict.keys())

    for cid in range(n_clients):
        dom_groups = client_dominant_groups[cid]
        dom_users = set()
        for g in dom_groups:
            dom_users.update(group_to_users[g])

        dom_users = list(dom_users)
        random.shuffle(dom_users)
        n_dom = int(dominant_ratio * len(dom_users))
        clients_users[cid].update(dom_users[:n_dom])

        # 混入非 dominant 的其他用户
        remaining_pool = list(all_users - clients_users[cid])
        random.shuffle(remaining_pool)
        n_other = int((1 - dominant_ratio) * len(dom_users))
        clients_users[cid].update(remaining_pool[:n_other])

    clients_data = {}
    for cid, uids in enumerate(clients_users):
        uids = list(uids)
        random.shuffle(uids)
        split_point = int(len(uids) * (1 - test_ratio))
        train_ids = uids[:split_point]
        test_ids = uids[split_point:]

        clients_data[cid] = {
            'train': {uid: user_dict[uid] for uid in train_ids},
            'test': {uid: user_dict[uid] for uid in test_ids},
        }

    return clients_data, client_dominant_groups


def save_clients_datasets(clients_data, group_id_to_idx, save_dir='../../Parsed_dataset/BlogCatalog'):
    os.makedirs(save_dir, exist_ok=True)

    for cid, splits in clients_data.items():
        # 合并 train 和 test
        all_users = dict()
        train_ids = set()
        test_ids = set()

        for uid, info in splits['train'].items():
            all_users[uid] = info
            train_ids.add(uid)

        for uid, info in splits['test'].items():
            all_users[uid] = info
            test_ids.add(uid)

        # 保留图内部边
        valid_uids = set(all_users.keys())
        filtered_users = {}
        for uid, info in all_users.items():
            filtered_following = [f for f in info.get('following', []) if f in valid_uids]
            new_info = dict(info)
            new_info['following'] = filtered_following
            filtered_users[uid] = new_info

        # 构建图
        data = preprocess_social_graph(filtered_users, group_id_to_idx)

        # 创建 mask
        uid_to_idx = {uid: i for i, uid in enumerate(data.user_ids.tolist())}
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        for uid in train_ids:
            if uid in uid_to_idx:
                train_mask[uid_to_idx[uid]] = True
        for uid in test_ids:
            if uid in uid_to_idx:
                test_mask[uid_to_idx[uid]] = True

        data.train_mask = train_mask
        data.test_mask = test_mask

        # 保存一份完整图 + mask 文件
        save_path = os.path.join(save_dir, f'client{cid}.pt')
        torch.save(data, save_path)
        print(f"[Saved] Client {cid} full graph with train/test masks to {save_path}")



def main():
    edge_file = '../../Dataset/BlogCatalog/BlogCatalog-dataset/data/edges.csv'
    node_file = '../../Dataset/BlogCatalog/BlogCatalog-dataset/data/group-edges.csv'
    user_dict = read_data(edge_file, node_file)
    group_id_to_idx = extract_global_group_map(user_dict)
    clients_data, dominant_groups = split_user_dict_with_domain_shift(
        user_dict, n_clients=3)

    # 可视化组分布
    for cid, split in clients_data.items():
        train_data = split['train']
        test_data = split['test']
        all_groups = []

        for u in train_data.values():
            all_groups.extend(u['groups'])
        for u in test_data.values():
            all_groups.extend(u['groups'])

        group_count = Counter(all_groups)

        print(f"\n=== Client {cid} ===")
        print(f"Dominant Groups: {dominant_groups[cid]}")
        print(f"Train Samples: {len(train_data)} | Test Samples: {len(test_data)}")


    # 保存
    save_clients_datasets(clients_data, group_id_to_idx)


if __name__ == "__main__":
    main()






