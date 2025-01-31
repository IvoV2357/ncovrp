from torch.utils.data import Dataset
import torch
import numpy as np
import os
from typing import NamedTuple
import torch.nn.functional as F

def _pad_mask(mask):
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


def _mask_bool2byte(mask):
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def _mask_byte2long(mask):
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(-1)


def mask_bool2long(mask):
    return _mask_byte2long(_mask_bool2byte(mask))


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(*mask.size()[:-1], -1)[..., :n]

def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0

def mask_long2bool(mask, n=None):
    return _mask_byte2bool(_mask_long2byte(mask), n=n)

def mask_long_scatter(mask, values, check_unset=True):
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None] 
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    return mask | (where.long() << (values_ % 64))

def read_instance(file):
    instance = {}
    with open(file=file, mode='r') as f:
        lines = f.readlines()
        current_section = None
        coordinates = {}
        demands = {}
        coordmax = 0
        capacity = 0
        for line in lines:
            if line.startswith('CAPACITY'):
                capacity = int(line.split(':')[1])
            if line.startswith('NODE_COORD_SECTION'):
                current_section = 'NODE_COORD_SECTION'
                continue
            elif line.startswith('DEMAND_SECTION'):
                current_section = 'DEMAND_SECTION'
                continue
            elif line.startswith('DEPOT_SECTION'):
                current_section = None
            if current_section == 'NODE_COORD_SECTION':
                node, x, y = map(int, line.split())
                coordmax = max(coordmax, x, y)
                coordinates[node] = (x, y)
            if current_section == 'DEMAND_SECTION':
                node, demand = map(int, line.split())
                demands[node] = demand
        if coordmax < 1000:
            coordmax = 1000
        instance['depot'] = torch.tensor(coordinates[1])
        coords = np.array([coordinates[i] for i in range(2, len(coordinates)+1)])
        coords = coords / coordmax
        demand = np.array([demands[i] for i in range(2, len(demands)+1)])
        demand = demand / capacity
        instance['demand'] = torch.from_numpy(demand).float()
        instance['coordinates'] = torch.from_numpy(coords).float()
        
    return instance

class CVRPDataset(Dataset):
    def __init__(self, generate=True, samples=1024, n=50):
        super(CVRPDataset, self).__init__()
        if generate:
            self.size = samples
            self.capacity = 40
            self.data = []
            for i in range(samples):
                demands = np.random.randint(1, 10, n)
                coordinates = np.random.rand(n, 2)
                depot = np.random.rand(2)
                demands = demands / self.capacity
                instance = {
                    'depot': torch.tensor(depot).float(),
                    'demand': torch.tensor(demands).float(),
                    'coordinates': torch.tensor(coordinates).float()
                }
                self.data.append(instance)
        else:
            self.data = []
            for file in os.listdir('./data'):
                if file.endswith('.vrp'):
                    instance = read_instance(f'./data/{file}')
                    self.data.append(instance)
            self.size = len(self.data)
    
    def get_costs(self, dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        sorted_pi = pi.data.sort(1)[0]

        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -self.capacity),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i] 
            used_cap[used_cap < 0] = 0


        coordinates_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['coordinates']), 1)
        d = coordinates_with_depot.gather(1, pi[..., None].expand(*pi.size(), coordinates_with_depot.size(-1)))

        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  
        ), None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)
    
    
class StateCVRP(NamedTuple):
    coords: torch.Tensor 
    demand: torch.Tensor
    ids: torch.Tensor  
    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  
    VEHICLE_CAPACITY = 1.0 

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        coordinates = input['coordinates']
        demand = input['demand']

        batch_size, n_coordinates, _ = coordinates.size()
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], coordinates), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=coordinates.device)[:, None],  
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=coordinates.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(
                torch.zeros(
                    batch_size, 1, n_coordinates + 1,
                    dtype=torch.uint8, device=coordinates.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_coordinates + 63) // 64, dtype=torch.int64, device=coordinates.device) 
            ),
            lengths=torch.zeros(batch_size, 1, device=coordinates.device),
            cur_coord=input['depot'][:, None, :], 
            i=torch.zeros(1, dtype=torch.int64, device=coordinates.device) 
        )

    def get_final_cost(self):
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        selected = selected[:, None]  
        prev_a = selected
        n_coordinates = self.demand.size(-1)  
        cur_coord = self.coords[self.ids, selected]

        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_coordinates - 1)]

        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        if self.visited_.dtype == torch.uint8:
            visited_coordinates = self.visited_[:, :, 1:]
        else:
            visited_coordinates = mask_long2bool(self.visited_, n=self.demand.size(-1))
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        mask_coordinates = visited_coordinates.to(exceeds_cap.dtype) | exceeds_cap
        mask_depot = (self.prev_a == 0) & ((mask_coordinates == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_coordinates), -1)

    def construct_solutions(self, actions):
        return actions