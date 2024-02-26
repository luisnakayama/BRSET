import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=50, backbone='Retina', save=False, device='cpu', patience=7):
    model.to(device)

    binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,
    }
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)

        for batch in tqdm(train_dataloader, total=num_train_batches):
            inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if binary:
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, torch.argmax(labels, dim=1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_train_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                val_inputs = val_batch['image'].to(device)
                val_labels = val_batch['labels'].to(device)

                val_outputs = model(val_inputs)

                if binary:
                    val_loss += criterion(val_outputs, val_labels.float()).item()
                else:
                    val_loss += criterion(val_outputs, torch.argmax(val_labels, dim=1)).item()

                preds = torch.argmax(val_outputs, dim=1) if not binary else val_outputs.round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(torch.argmax(val_labels, dim=1).cpu().numpy() if not binary else val_labels.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_scores.append(f1)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, F1 Score: {f1}')

        if f1 > best_model_info['f1_score']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

    if save:
        os.makedirs('Models', exist_ok=True)
        model.load_state_dict(best_model_info['state_dict'])
        torch.save(model.state_dict(), f'Models/fine_tuned_{backbone}_best.pth')

    return model
