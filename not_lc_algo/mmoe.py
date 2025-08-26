import torch
import torch.nn as nn

class MMOE(nn.Module):
    def __init__(self, input_dim, n_experts, n_tasks, expert_hidden_dim=(64,32), tower_hidden_dim=(32,), gate_hidden_dim=(16,)):
        super(MMOE, self).__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim[0]),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim[0], expert_hidden_dim[1]),
                nn.ReLU()                
            ) for _ in range(self.n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_hidden_dim[0]),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim[0], n_experts),
                nn.Softmax(dim=-1)              
            ) for _ in range(self.n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim[1], tower_hidden_dim[0]),
                nn.ReLU(),
                nn.Linear(tower_hidden_dim[0], 1),
            ) for _ in range(self.n_tasks)
        ])

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_outputs = []
        for i in range(self.n_tasks):
            gate_output = self.gates[i](x)
            gate_output = gate_output.unsqueeze(1)
            weight_expert_output = torch.matmul(gate_output, expert_outputs)
            weight_expert_output = weight_expert_output.squeeze(1)
            tower_output = self.towers[i](weight_expert_output)
            task_outputs.append(tower_output)
        return task_outputs