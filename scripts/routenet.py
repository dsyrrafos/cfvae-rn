import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    

class RouteNet(nn.Module):
    def __init__(self, input_dim_path=1, input_dim_link=1, output_dim=1, iterations=4, link_state_dim=64, path_state_dim=64, aggr_dropout=0, readout_dropout=0, device='cpu'):
        super(RouteNet, self).__init__()

        self.input_dim_path = input_dim_path
        self.input_dim_link = input_dim_link
        self.output_dim = output_dim
        self.iterations = iterations
        self.link_state_dim = link_state_dim
        self.path_state_dim = path_state_dim
        self.aggr_dropout = aggr_dropout
        self.readout_dropout = readout_dropout
        self.device = device

        self.link_update = nn.GRUCell(self.path_state_dim, self.link_state_dim)
        self.path_update = nn.GRU(self.link_state_dim, self.path_state_dim, batch_first=True)

        self.path_embedding = nn.Sequential(
            nn.Linear(self.input_dim_path, self.path_state_dim),
            nn.ReLU(),
            nn.Linear(self.path_state_dim, self.path_state_dim),
            nn.ReLU()
        )

        self.link_embedding = nn.Sequential(
            nn.Linear(self.input_dim_link, self.link_state_dim),
            nn.ReLU(),
            nn.Linear(self.link_state_dim, self.link_state_dim),
            nn.ReLU()
        )

        self.aggr_mlp = nn.Sequential(
            nn.Linear(4 * self.path_state_dim, 2 * self.path_state_dim),
            nn.ReLU(),
            nn.Dropout(self.aggr_dropout),
            nn.Linear(2 * self.path_state_dim, 2 * self.path_state_dim),
            nn.ReLU(),
            nn.Dropout(self.aggr_dropout),
            nn.Linear(2 * self.path_state_dim, self.path_state_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        traffic = inputs['traffic'].float().to(self.device)
        capacity = inputs['capacity'].float().to(self.device)

        link_to_path = inputs['link_to_path'].to(self.device)
        path_to_link = inputs['path_to_link'].to(self.device)

        path_state = self.path_embedding(traffic)
        link_state = self.link_embedding(capacity)

        for iteration in range(self.iterations):
            # Gather links                
            expanded_links = link_state[0][link_to_path.clamp(min=0)]
            mask = (link_to_path != -1).unsqueeze(-1)  # mask padded positions
            link_gather = expanded_links * mask  # apply mask

            # Get sequence lengths
            lengths = mask.sum(dim=-2).squeeze(0).squeeze(-1).to(torch.long)

            # GRU update
            packed_input = pack_padded_sequence(link_gather.squeeze(0), lengths.cpu(), batch_first=True, enforce_sorted=False)

            old_path_state = path_state

            path_state_sequence, path_state = self.path_update(packed_input, old_path_state)

            output_padded, _ = pad_packed_sequence(path_state_sequence, batch_first=True)

            # Concatenate old state
            path_state_sequence = torch.cat([old_path_state.squeeze(0).unsqueeze(1), output_padded], dim=1)

            # Gather path states per link
            expanded_paths = path_state_sequence[path_to_link[..., 0].clamp(min=0), path_to_link[..., 1].clamp(min=0)]
            mask_paths = (path_to_link[..., 0] != -1).unsqueeze(-1)
            path_gather = expanded_paths * mask_paths
            # lengths_paths = mask_paths.sum(dim=-2).squeeze(0).squeeze(-1).to(torch.long)

            # Aggregations
            path_min = path_gather.masked_fill(~mask_paths, float('inf')).min(dim=-2).values
            path_max = path_gather.masked_fill(~mask_paths, float('-inf')).max(dim=-2).values
            path_sum = path_gather.masked_fill(~mask_paths, 0).sum(dim=-2)
            valid_counts = mask_paths.sum(dim=-2).clamp(min=1)
            path_mean = path_sum / valid_counts

            # MLP aggregation and link update
            aggregation = torch.cat([path_min, path_max, path_sum, path_mean], dim=2)
            path_aggregation = self.aggr_mlp(aggregation)
            link_state = self.link_update(path_aggregation.squeeze(0), link_state.squeeze(0)).unsqueeze(0)

        return path_state, link_state


class DeterministicDecoder(nn.Module):
    def __init__(self, path_state_dim, output_dim, readout_dropout):
        super().__init__()
        
        hidden = nn.Sequential(
            nn.Linear(path_state_dim, path_state_dim // 2),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(path_state_dim // 2, path_state_dim // 4),
            nn.ReLU(),
            nn.Dropout(readout_dropout)
        )

        self.hidden = hidden
        self.output_layer = nn.Linear(path_state_dim // 4, output_dim)

    def forward(self, x):
        h = self.hidden(x)
        out = self.output_layer(h)
        return out.squeeze()


class RouteNetPredictor(nn.Module):
    def __init__(self, input_dim_path=1, input_dim_link=1, output_dim=1, iterations=4, link_state_dim=64, path_state_dim=64, aggr_dropout=0, readout_dropout=0, device='cpu'):
        super(RouteNetPredictor, self).__init__()
        self.device = device

        self.routenet = RouteNet(
            input_dim_path,
            input_dim_link,
            output_dim,
            iterations,
            link_state_dim,
            path_state_dim,
            aggr_dropout,
            readout_dropout,
            device
        )

    def forward(self, inputs):
        path_state, _ = self.routenet(inputs)
        r = self.readout(path_state)
        return r, path_state
    

class RouteNetEncoder(nn.Module):
    def __init__(self, input_dim_path=2, input_dim_link=1, output_dim=1, iterations=4, link_state_dim=64, path_state_dim=64, aggr_dropout=0, readout_dropout=0, device='cpu'):
        super(RouteNetEncoder, self).__init__()

        self.device = device
        self.routenet = RouteNet(
            input_dim_path,
            input_dim_link,
            output_dim,
            iterations,
            link_state_dim,
            path_state_dim,
            aggr_dropout,
            readout_dropout,
            device
        )

    def forward(self, data):
        inputs, targets = data
        traffic = inputs['traffic'].float().to(self.device)
        capacity = inputs['capacity'].float().to(self.device)
        targets_list = [targets[key].unsqueeze(2).float().to(self.device) for key in targets.keys()]
        path_attributes = torch.cat([traffic] + targets_list, dim=2)  # shape: [1, n_paths, num_targets + 1]
        augmented_inputs = {'traffic': path_attributes, 'capacity': capacity, 'link_to_path': inputs['link_to_path'], 'path_to_link': inputs['path_to_link']}
        path_state, link_state = self.routenet(augmented_inputs)
        return path_state.squeeze(0), link_state.squeeze(0)


class RouteNetPriorEncoder(nn.Module):
    def __init__(self, input_dim_path=1, input_dim_link=1, output_dim=1, iterations=4, link_state_dim=64, path_state_dim=64, aggr_dropout=0, readout_dropout=0, device='cpu'):
        super(RouteNetPriorEncoder, self).__init__()

        self.routenet = RouteNet(
            input_dim_path,
            input_dim_link,
            output_dim,
            iterations,
            link_state_dim,
            path_state_dim,
            aggr_dropout,
            readout_dropout,
            device
        )

    def forward(self, inputs):
        path_state, link_state = self.routenet(inputs)
        return path_state.squeeze(0), link_state.squeeze(0)

