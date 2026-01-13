import numpy as np
import networkx as nx
from tqdm import tqdm

from torch.utils.data import Dataset, ConcatDataset, DataLoader, SubsetRandomSampler

from datanetAPI import DatanetAPI

class DatanetDataset(Dataset):
    def __init__(self, data_dir, filenames, size, indices=None, shuffle=False):
        self.data_dir = data_dir
        self.filenames = filenames
        self.size = size
        self.shuffle = shuffle
        self.samples = list(self.generator()) #TODO: Lazy loading
        self.indices = indices if indices is not None else range(len(self.samples))

    def generator(self):
        tool  = DatanetAPI(self.data_dir, self.shuffle)
        counter = 0
    
        for i, sample in enumerate(iter(tool)):
            f = sample._get_data_set_file_name()
            print(f"{i}: File: {f}")

            if f not in self.filenames:
                print(f"File {f} not in the list of files to process")
                continue

            counter += 1

            # Check if the sample is empty
            if self.size is not None and counter > self.size:
                break

            G_copy = sample.get_topology_object()
            T = sample.get_traffic_matrix()
            R = sample.get_routing_matrix()
            D = sample.get_performance_matrix()

            HG = self.network_to_hypergraph(network_graph=G_copy,
                                    routing_matrix=R,
                                    traffic_matrix=T,
                                    performance_matrix=D)

            ret = self.hypergraph_to_input_data(HG)

            yield ret

    def network_to_hypergraph(self, network_graph, routing_matrix, traffic_matrix, performance_matrix):
        G = nx.DiGraph(network_graph)
        R = routing_matrix
        T = traffic_matrix
        D = performance_matrix

        D_G = nx.DiGraph()
        for src in range(G.number_of_nodes()):
            for dst in range(G.number_of_nodes()):
                if src != dst:
                    if G.has_edge(src, dst):
                        D_G.add_node('l_{}_{}'.format(src, dst),
                                    capacity=float(G.edges[src, dst]['bandwidth']))

                    avg_bw = T[src, dst]['Flows'][0]['AvgBw']
                    pkts_gen = T[src, dst]['Flows'][0]['PktsGen']
                    if avg_bw != 0 and pkts_gen != 0:
                        D_G.add_node('p_{}_{}'.format(src, dst),
                                        traffic=T[src, dst]['Flows'][0]['AvgBw'],
                                        packets=T[src, dst]['Flows'][0]['PktsGen'],
                                        delay=D[src, dst]['Flows'][0]['AvgDelay'],
                                        jitter=D[src, dst]['Flows'][0]['Jitter'],
                                        loss=D[src, dst]['Flows'][0]['PktsDrop'])

                        for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                            D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                            D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}'.format(src, dst))

        D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

        return D_G

    def hypergraph_to_input_data(self, hypergraph):
        n_p = 0
        n_l = 0
        mapping = {}
        for entity in list(hypergraph.nodes()):
            if entity.startswith('p'):
                mapping[entity] = ('p_{}'.format(n_p))
                n_p += 1
            elif entity.startswith('l'):
                mapping[entity] = ('l_{}'.format(n_l))
                n_l += 1

        G = nx.relabel_nodes(hypergraph, mapping)

        link_to_path = []
        path_to_link = []
        for node in G.nodes:
            in_nodes = [s for s, d in G.in_edges(node)]
            if node.startswith('l_'):
                touple = []
                for n in in_nodes:
                    path_pos = [d for s, d in G.out_edges(n)]
                    touple.append([int(n.replace('p_', '')), path_pos.index(node)])
                path_to_link.append(touple)
            if node.startswith('p_'):
                link_to_path.append([int(n.replace('l_', '')) for n in in_nodes])

        return {"traffic": np.expand_dims(list(nx.get_node_attributes(G, 'traffic').values()), axis=1),
                "capacity": np.expand_dims(list(nx.get_node_attributes(G, 'capacity').values()), axis=1),
                "link_to_path": link_to_path, "path_to_link": path_to_link
                }, {"delay": list(nx.get_node_attributes(G, 'delay').values()),
                "jitter": list(nx.get_node_attributes(G, 'jitter').values()),
                "loss": list(nx.get_node_attributes(G, 'loss').values())
                }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # real_idx = self.indices[idx]
        return self.samples[idx]

    def standarize(self, key, mean, std, target=False):
        """
        Standardize a feature by (value - mean) / std.
        
        Args:
            key (str): The dictionary key to normalize.
            mean (float or np.ndarray): Mean value(s).
            std (float or np.ndarray): Std value(s).
            target (bool): If True, apply normalization to y (target dict),
                        otherwise apply to x (input dict).
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if target:
                if key in y:
                    y[key] = (np.array(y[key]) - mean) / std
            else:
                if key in x:
                    x[key] = (np.array(x[key]) - mean) / std
            self.samples[i] = (x, y)
        return self
    
    def destandarize(self, key, mean, std, target=False):
        """
        Revert standardization: x = x * std + mean

        Args:
            key (str): The dictionary key to revert.
            mean (float or np.ndarray): Mean value(s) used for normalization.
            std (float or np.ndarray): Std value(s) used for normalization.
            target (bool): If True, apply to y (target dict),
                        otherwise apply to x (input dict).
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if target:
                if key in y:
                    y[key] = np.array(y[key]) * std + mean
            else:
                if key in x:
                    x[key] = np.array(x[key]) * std + mean
            self.samples[i] = (x, y)
        return self

    def log_transform(self, key, target=True):
        """
        Apply log transformation to a feature: x = log(x + 1)

        Args:
            key (str): The dictionary key to transform.
            target (bool): If True, apply to y (target dict),
                        otherwise apply to x (input dict).
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if target:
                if key in y:
                    y[key] = np.log(np.array(y[key]) + 1)
            else:
                if key in x:
                    x[key] = np.log(np.array(x[key]) + 1)
            self.samples[i] = (x, y)
        return self
    
    def exp_transform(self, key, target=True):
        """
        Revert log transformation: x = exp(x) - 1

        Args:
            key (str): The dictionary key to revert.
            target (bool): If True, apply to y (target dict),
                        otherwise apply to x (input dict).
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if target:
                if key in y:
                    y[key] = np.exp(np.array(y[key])) - 1
            else:
                if key in x:
                    x[key] = np.exp(np.array(x[key])) - 1
            self.samples[i] = (x, y)
        return self
    
    def list_to_numpy(self, key, target=True):
        """
        Transform list to numpy array for a feature.

        Args:
            key (str): The dictionary key to transform.
            target (bool): If True, apply to y (target dict),
                        otherwise apply to x (input dict).
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if target:
                if key in y:
                    y[key] = np.array(y[key])
            else:
                if key in x:
                    x[key] = np.array(x[key])
            self.samples[i] = (x, y)
        return self
    
    def keep_only_targets(self, keys):
        """
        Keep only specified keys in the target dictionary.

        Args:
            keys (list of str): List of keys to keep in the target dict.
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            y = {k: v for k, v in y.items() if k in keys}
            self.samples[i] = (x, y)
        return self
    
    def pad_sequences(self, key, max_length, padding_value=-1):
        """
        Pad sequences to a maximum length.

        Args:
            key (str): The dictionary key to pad.
            max_length (int): The maximum length to pad sequences to.
            padding_value (int, float): The value to use for padding.
        """
        for i in tqdm(range(len(self.samples))):
            x, y = self.samples[i]
            if key in x:
                samples_list = []
                samples_lengths = []
                for sample in x[key]:
                    seq = np.array(sample)
                    if key == 'path_to_link':
                        padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant', constant_values=padding_value)
                    elif key == 'link_to_path':
                        padded_seq = np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=padding_value)
                    else:
                        raise ValueError(f"Unsupported key for padding: {key}")
                    samples_list.append(padded_seq)
                    samples_lengths.append(len(padded_seq))
                x[key] = np.stack(samples_list, axis=0)
        self.samples[i] = (x, y)
        return self
    
    def get_max_sequence_length(self, key):
        """
        Get the maximum sequence length for a feature.

        Args:
            key (str): The dictionary key to check.
        """
        max_length = 0
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if key in x:
                for sample in x[key]:
                    seq_length = len(sample)
                    if seq_length > max_length:
                        max_length = seq_length
        return max_length

    # Perturbation methods can be added here as needed
    def perturb_feature(self, key, std, noise_level=0.1, target=False):
        """
        Perturb a feature by adding Gaussian noise.

        Args:
            key (str): The dictionary key to perturb.
            noise_level (float): Standard deviation of the Gaussian noise.
            target (bool): If True, apply to y (target dict),
                        otherwise apply to x (input dict).
        """
        for i in range(len(self.samples)):
            x, y = self.samples[i]
            if target:
                if key in y:
                    noise = np.random.normal(0, noise_level*std, size=np.array(y[key]).shape)
                    y[key] = np.array(y[key]) + noise
            else:
                if key in x:
                    noise = np.random.normal(0, noise_level*std, size=np.array(x[key]).shape)
                    x[key] = np.array(x[key]) + noise
            self.samples[i] = (x, y)
        return self


class DatanetConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(DatanetConcatDataset, self).__init__(datasets)
    
    def get_max_sequence_length(self, key):
        max_length = 0
        for dataset in self.datasets:
            dataset_max_length = dataset.get_max_sequence_length(key)
            if dataset_max_length > max_length:
                max_length = dataset_max_length
        return max_length
    
    def pad_sequences(self, key, max_length, padding_value=-1):
        for i, dataset in enumerate(self.datasets):
            print(f"Padding dataset {i+1} out of {len(self.datasets)}")
            dataset.pad_sequences(key, max_length, padding_value)
        return self


def get_dataloader_with_sampling(dataset, dataset_type, dataset_subset_size, shuffle=False, batch_size=1, num_workers=0):
    if dataset_type == 'routenet':
        if shuffle:
            subset_indices = np.random.choice(len(dataset), dataset_subset_size, replace=False)
            sampler = SubsetRandomSampler(subset_indices)
            return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
        else:
            return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

