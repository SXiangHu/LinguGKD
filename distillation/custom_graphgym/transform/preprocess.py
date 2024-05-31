import torch
import pickle
from torch_geometric.graphgym.config import cfg

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
def integrate_features(map_count):
    for node, item in map_count.items():
        map_count[node]['llm_features_correct'] = {}
        # values = list(item['features'].values())
        for hop in range(4):
            hop = str(hop)
            features = item['features'][hop]
            if len(features) == 0:
                if hop == '0':
                    break
                else:
                    map_count[node]['llm_features_correct'][hop] = map_count[node]['llm_features_correct'][str(
                        int(hop)-1)]
                    continue
            if isinstance(features, list):
                map_count[node]['llm_features_correct'][hop] = torch.mean(
                    torch.stack(features), dim=0)
            else:
                map_count[node]['llm_features_correct'][hop] = features
    return map_count

def get_preprocess_func(data_dir, llm_dir):
    def preprocess_func(data):
        # load llm features
        mode = 'full'
        hops = cfg.gnn.layers_mp
        feature_key = 'llm_features_full' if mode == 'full' else 'llm_features_correct'
        # 加载 llm_features
        llm_features = load_pkl(llm_dir + f'/llm_features_train.pkl')
        llm_features_test = load_pkl(llm_dir + f'/llm_features_test.pkl')
        # llm_features_test = integrate_features(llm_features_test)
            
        # load train_ids
        # 计算 llm_features 所有 value 的 keys 的交集
        # train_ids = list(set.intersection(
        #     *[set(v.keys()) for v in llm_features]))
        # train_ids = list(llm_features.keys())
        train_ids = [key for key, item in llm_features.items()
                     if len(item.get(feature_key, [])) > 0]

        train_ids_full = load_pkl(data_dir + '/train_ids.pkl')
        ##########################
        # train_ids = train_ids_full
        ##########################
        partial = len(set(train_ids)) / len(set(train_ids_full))
        print('#'*20, f'len(train_ids): {len(train_ids)}, len(train_ids_full): {len(train_ids_full)}, {partial}', '#'*20)
        
        data._data.train_mask = torch.zeros(
            data._data.num_nodes, dtype=torch.bool)
        data._data.train_mask[train_ids] = 1
        # load test_ids
        test_ids_full = load_pkl(data_dir + '/test_ids.pkl')
        test_ids = test_ids_full
        # test_ids = [key for key, item in llm_features_test.items()
        #              if len(item['features']) == 4]
        data._data.test_mask = torch.zeros(
            data._data.num_nodes, dtype=torch.bool)
        data._data.test_mask[test_ids] = 1
        # set val_mask = test_mask
        data._data.val_mask = data._data.test_mask
        ####################################
        data._data.llm_feature = torch.randn(
            hops+1, data._data.num_nodes, cfg.gnn.dim_llm)
        for k in range(hops+1):
            for i in train_ids:
                data._data.llm_feature[k, i] = torch.tensor(llm_features[i][feature_key][str(
                    k)])
            for i in test_ids:
                data._data.llm_feature[k, i] = torch.tensor(llm_features_test[i]['llm_features_full'][str(
                    k)])
                # if i in llm_features_test:
                #     data._data.llm_feature[k, i] = llm_features_test[i]['features'][str(k)]
        #####################################
        return data
    return preprocess_func