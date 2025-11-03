import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from utils import load_client_data
from model import TabularMLP 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FLClient(fl.client.NumPyClient):
    """Flower client for PyTorch model, using class weights for imbalance."""

    def __init__(self, model, train_loader, test_loader, device, class_weights): 
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        # Create the weighted loss function here
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(self.device)) # <-- WEIGHTS APPLIED

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        lr = float(config.get("lr", 1e-3))
        local_epochs = int(config.get("local_epochs", 1))
        
        # --- CORRECTION APPLIED HERE ---
        # The optimizer MUST be initialized with self.model.parameters(), not just 'model.parameters()'
        # The 'model' variable is undefined in the scope of this method.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # --- END CORRECTION ---

        total_loss, total_correct, total_samples = 0.0, 0, 0

        for _ in range(local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                
                loss = self.loss_fn(out, y) 
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X.size(0)
                total_correct += (out.argmax(dim=1) == y).sum().item()
                total_samples += X.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return self.get_parameters(), total_samples, {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, n = 0.0, 0, 0

        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                
                batch_loss = self.loss_fn(out, y).item()
                
                loss += batch_loss * y.size(0)

                probs = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                correct += (preds == y).sum().item()
                n += y.size(0)

        # Compute metrics
        accuracy = correct / n
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            # Handle case where only one class is present in the test batch
            auc = 0.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

        # NOTE: Loss returned here is WEIGHTED loss because self.loss_fn is weighted
        return float(loss / n), n, metrics 


def start_client(csv_path, server_address="127.0.0.1:8080", batch_size=32, local_epochs=1):
    train_ds, test_ds, in_dim, class_weights = load_client_data(csv_path) 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    output_dim = 2
    
    # Model instantiated here
    model = TabularMLP(input_dim=in_dim, output_dim=output_dim)

    client = FLClient(model, train_loader, test_loader, DEVICE, class_weights)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()
    start_client(args.data, args.server, args.batch_size, args.local_epochs)
