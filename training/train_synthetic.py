import argparse

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from uncertainty_cfs.architectures import MLP
from uncertainty_cfs.data import load_datasets
from uncertainty_cfs.decompositions import (
    entropy_based_uncertainty_quantification,
    variance_based_uncertainty_quantification,
)
import numpy as np
import torch
import torch.nn as nn
from probly.representation import Ensemble, Dropout, Bayesian
from probly.losses import ELBOLoss
from probly.calibration import Temperature
from probly.metrics import expected_calibration_error

from uncertainty_cfs.data import (
    load_l_dataset,
    load_ring_dataset,
    load_bubbles,
    load_bubbles_noisy,
    load_one_moon,
    load_two_moon,
    load_infinity_dataset,
)

### Parse arguments ###
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="dropout",
    help="Model to use (default: dropout). Can be 'dropout', 'bayesian', or 'ensemble'",
)
parser.add_argument(
    "--n_mc_samples",
    type=int,
    default=50,
    help="Number of Monte Carlo samples (default: 50)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size (default: 128)",
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=100,
    help="Number of epochs (default: 100)",
)
parser.add_argument(
    "--n_models",
    type=int,
    default=20,
    help="Number of models in the ensemble (default: 20)",
)
parser.add_argument(
    "--ensemble_type",
    type=str,
    default="deep",
    help="Type of the ensemble trainng (default: deep_ensemble). Can be 'deep_ensemble', 'dare' or 'adversarial' ",
)
args = parser.parse_args()


torch._functorch.config.donated_buffer = False

class RegularizerDare(nn.Module):
    def __init__(self, lambda_reg=0.01):
        super(RegularizerDare, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, model):
        total_loss = 0
        for param in model.parameters():
            total_loss += param.square().log2().sum()
        return total_loss * self.lambda_reg
    

if __name__ == "__main__":
    data_files = [
        "one_moon",
        "two_moon",
        "infinity",
        "ring",
        "bubbles",
        "bubbles_noisy",
        "l_dataset",
    ]
    for dataset_name in data_files:
        torch.manual_seed(42)
        np.random.seed(42)
        ### Config ###
        DATASET_NAME = dataset_name
        MODEL_NAME = args.model  # "dropout"  # "ensemble"  # "bayesian"
        N_MC_SAMPLES = args.n_mc_samples
        BATCH_SIZE = args.batch_size
        N_EPOCHS = args.n_epochs
        DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### Load Data ###
        
        X,y,_ = load_datasets(dataset_name)
        X_train, X_test, y_train, y_test = X,X,y,y
        
        X_train,X_cal,y_train,y_cal = train_test_split(X_train,y_train,test_size=0.1,random_state=42,stratify=y_train)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_cal_tensor = torch.tensor(y_cal, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        print("Proportion of classes in train set: ", np.bincount(y_train))
        print("Proportion of classes in cal set: ", np.bincount(y_cal))
        print("Proportion of classes in test set: ", np.bincount(y_test))

        ### Build DataLoader ###
        train_dataset = torch.utils.data.TensorDataset(
            X_train_tensor,
            y_train_tensor,
        )
        cal_dataset = torch.utils.data.TensorDataset(
            X_cal_tensor,
            y_cal_tensor,
        )
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        calloader = torch.utils.data.DataLoader(
            cal_dataset, batch_size=BATCH_SIZE
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        ### Build Model ###
        architecture = MLP(
            input_dim=X_train.shape[1],
            output_dim=len(np.unique(y_train)),
            hidden_dims=[64, 64],
            batch_norm=False,
        )
        ### Setup Model ###
        if MODEL_NAME == "ensemble":
            model = Ensemble(architecture, n_members=args.n_models)
        elif MODEL_NAME == "bayesian":
            model = Bayesian(architecture)
        elif MODEL_NAME == "dropout":
            model = Dropout(architecture, p=0.2)
        ### Train Model ###
        if MODEL_NAME != "ensemble":
            training_models = [model]
        else:
            training_models = model.models
        optimizers = [
            torch.optim.Adam(m.parameters(), lr=1e-4) for m in training_models
        ]
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=N_EPOCHS, eta_min=1e-6
            ) for opt in optimizers
        ]

        if MODEL_NAME != "bayesian":
            criterion = nn.CrossEntropyLoss(weight=None)
        else:
            criterion = ELBOLoss(weight=None, kl_penalty=1e-5)
            
        if MODEL_NAME == "ensemble" and args.ensemble_type == "dare":
            regularizer = RegularizerDare(lambda_reg=0.01)
        print(f"Training {MODEL_NAME} on {dataset_name}...")
        ##### Train the model ####

        for net, optimizer, scheduler in zip(
            training_models, optimizers, schedulers
        ):
            net.to(DEVICE)
            for epoch in range(N_EPOCHS):
                net.train()
                total_loss = 0
                batches_X = []
                batches_y = []
                for batch_X, batch_y in train_loader:
                    batches_X.append(batch_X)
                    batches_y.append(batch_y)
                    # Check class imbalance in the batch
                    # unique, counts = torch.unique(batch_y, return_counts=True)
                    # class_counts = dict(zip(unique.tolist(), counts.tolist()))
                    # print(f"Class distribution in batch: {class_counts}")
                    # Move data to device
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    if MODEL_NAME == "ensemble" and args.ensemble_type == "adversarial":
                        # Enable gradients for input features
                        batch_X.requires_grad = True
                    ### Forward pass ###    
                    optimizer.zero_grad()
                    outputs = net(batch_X)
                    #print("OUTPUTS: ", outputs[:10])
                    
                    ### Compute loss ###
                    
                    if MODEL_NAME == "bayesian":
                        loss = criterion(outputs, batch_y, net.kl_divergence)
                    else:
                        loss = criterion(outputs, batch_y)
                        
                    if MODEL_NAME == "ensemble" and args.ensemble_type == "dare":
                        # Gradually increase the regularization strength over the first 10 epochs
                        #after_10_epoch = epoch > 10
                        #lbmda = min(1.0, epoch / 10)
                        loss -= 0.01 * regularizer(net)
                        
                    if MODEL_NAME == "ensemble" and args.ensemble_type == "adversarial":
                        # Compute gradients w.r.t. inputs for FGSM
                        loss.backward(retain_graph=True)
                        # Feature-wise epsilon calculation
                        epsilon = 0.5 * torch.std(batch_X, dim=0, keepdim=True)
                        
                        # Generate adversarial examples using FGSM with feature-wise epsilon
                        adversarial_input = batch_X + epsilon * batch_X.grad.sign()
                        
                        # Clamp adversarial inputs to valid range (adjust bounds as needed)
                        adversarial_input = torch.clamp(adversarial_input, min=batch_X.min(), max=batch_X.max())

                        # Clear input gradients before next forward pass
                        batch_X.grad.zero_()
                        
                        # Forward pass with adversarial examples
                        outputs_adv = net(adversarial_input)
                        loss_adv = criterion(outputs_adv, batch_y)
                        
                        # Total loss (clean + adversarial)
                        clean_adv_equal_loss = 0.5 * loss + 0.5 * loss_adv
                        
                        # Backward pass for the combined loss
                        clean_adv_equal_loss.backward()
                        
                        # Update loss to be logged
                        loss = clean_adv_equal_loss
                    else:
                        loss.backward()
                    
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
                
                outputs = torch.empty(0, device=DEVICE)
                targets = torch.empty(0, device=DEVICE)

                for batch_X, batch_y in zip(batches_X, batches_y):
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    output = net(batch_X)
                    outputs = torch.cat((outputs, output), dim=0)
                    targets = torch.cat((targets, batch_y), dim=0)
                    
                ### Compute metrics ###
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.argmax(probs, dim=1)
                correct = (preds == targets).float()
                accuracy = correct.sum() / len(correct)
                recall = torch.zeros(2).to(DEVICE)
                precision = torch.zeros(2).to(DEVICE)
                f1 = torch.zeros(2).to(DEVICE)
                for cls in range(2):
                    true_positives = ((preds == cls) & (targets == cls)).sum().float()
                    false_positives = ((preds == cls) & (targets != cls)).sum().float()
                    false_negatives = ((preds != cls) & (targets == cls)).sum().float()
                    if (true_positives + false_negatives) > 0:
                            recall[cls] = true_positives / (true_positives + false_negatives)
                    if (true_positives + false_positives) > 0:
                            precision[cls] = true_positives / (true_positives + false_positives)
                    if (precision[cls] + recall[cls]) > 0:
                            f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
                print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")
                print(f"Recall per class: {recall.cpu().numpy()}")
                print(f"Precision per class: {precision.cpu().numpy()}")
                print(f"F1-score per class: {f1.cpu().numpy()}")
                
                ### Step the scheduler ###
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                else:
                    scheduler.step(avg_loss)    

        ### Calibrate Model ###
        temperature = Temperature(model)
        # temperature.to(DEVICE)
        temperature.fit(calloader, learning_rate=0.01, max_iter=100)
        print(f"Optimal temperature: {temperature.temperature.item():.4f}")
        # Update model to be the temperature scaled model
        model = temperature

        ### Save Model ###
        if MODEL_NAME == "ensemble":
            MODEL_NAME = f"{args.ensemble_type}_{MODEL_NAME}"
        torch.save(
            model.state_dict(),
            f"models/model={MODEL_NAME}_dataset={DATASET_NAME}.pth",
        )
            
        #### Evaluate on test set ####
        model.eval()
        with torch.no_grad():
            outputs = torch.empty(0, device=DEVICE)
            targets = torch.empty(0, device=DEVICE)
            for inpt, target in tqdm(test_loader):
                outputs = torch.cat((outputs, model.predict_pointwise(inpt.to(DEVICE),n_samples=50)), dim=0)
                targets = torch.cat((targets, target.to(DEVICE)), dim=0)
        correct = torch.sum(torch.argmax(outputs, dim=1) == targets).item()
        total = targets.size(0)
        ece = expected_calibration_error(outputs.cpu().numpy(), targets.cpu().numpy(), num_bins=10)
       # print(f"Softmax temperature: {model.temperature.item()}")
        print(f"Accuracy: {correct / total}")
        print(f"Expected Calibration Error: {ece}")

        model.load_state_dict(
            torch.load(f"models/model={MODEL_NAME}_dataset={DATASET_NAME}.pth")
        )
        print(model)
        if MODEL_NAME == "ensemble":
            preds = model.predict_representation(
                X_test_tensor
            )
        else:
            preds = model.predict_representation(X_test_tensor, n_samples=N_MC_SAMPLES)
        # print("Predicted probabilities (first 10 samples): ", preds[:10])
        te_e, au_e, eu_e = entropy_based_uncertainty_quantification(preds)
        te_v, au_v, eu_v = variance_based_uncertainty_quantification(preds)
        print(f"Total Entropy (first 10 samples): {te_e[:10]}--{te_v[:10]}")
        print(
            f"Aleatoric Uncertainty (first 10 samples): {au_e[:10]}--{au_v[:10]}"
        )
        print(
            f"Epistemic Uncertainty (first 10 samples): {eu_e[:10]}--{eu_v[:10]}"
        )