import torch
from numpy import ndarray
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import pickle
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fastapi import FastAPI
from starlette.responses import RedirectResponse
import uvicorn
import numpy as np

app = FastAPI(
    title='Citation category prediction',
    description='Graph Convolutional  Networks'
)
dataset = Planetoid(root='gnn_deploy/', name='cora', transform=NormalizeFeatures())

data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = F.softmax(self.out(x), dim=1)
        return x


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post("/input values between 1 - 2707")
def predict(sample: int):
    """"""
    "Enter input between 0 - 2707"
    """"""
    sample = sample
    model = pickle.load(open('model.pkl', 'rb'))
    pred = model(data.x, data.edge_index)
    result = pred[sample].detach().cpu().numpy()
    print(result)
    max_index_row = np.argmax(result)
    if max_index_row == 0:
        return {"Category of paper": "Case_Based"}
    if max_index_row == 1:
        return {"Category of paper": "Genetic_Algorithms"}
    if max_index_row == 2:
        return {"Category of paper": "Neural_Networks"}
    if max_index_row == 3:
        return {"Category of paper": "Probabilistic_Methods"}
    if max_index_row == 4:
        return {"Category of paper": "Reinforcement_Learning"}
    if max_index_row == 5:
        return {"Category of paper": "Rule_Learning"}
    if max_index_row == 6:
        return {"Category of paper": "Theory"}
    else:
        return {"Category of paper": "Not found"}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080, debug=True)

# print(predict(2707))