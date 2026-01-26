import torch
from moospread.utils.mobo_utils.learning.utils import prepare_dom_data, compute_class_weight, load_batched_dom_data, train_nn
from moospread.utils.mobo_utils.learning.model import NeuralNet
import torch.nn as nn



def init_dom_nn_classifier(x, y, rel_map, dom, n_var, batch_size=32,
                           activation='relu', lr=0.001, weight_decay=0.00001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 2 * n_var
    hidden_size = 200
    num_hidden_layers = 3
    epochs = 20
    data = prepare_dom_data(x, y, rel_map, dom, data_kind='tensor', device=device)
    # print(f"Data shape: {data.shape}")
    # print("data[:, -1]", data[:, -1])
    weight = compute_class_weight(data[:, -1])
    # print(f"Class weights: {weight}")
    if weight is None:
        return None

    net = NeuralNet(input_size, hidden_size, num_hidden_layers,
                    activation=activation).to(device)

    weight = torch.tensor(weight, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_nn(data, load_batched_dom_data, net, criterion, optimizer, batch_size, epochs)
    # print("Net", net)
    return net

