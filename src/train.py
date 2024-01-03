import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=20, backbone='Retina', save=False, device='cpu'):

    model.to(device)

    binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False
    
    # Initialize lists to store loss values
    train_losses = []
    val_losses = []
    f1_scores = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)

        # Training phase
        for batch in tqdm(train_dataloader,total=num_train_batches):

            # Move inputs and labels to the GPU if available
            inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)

            # Calculate the outputs
            outputs = model(inputs)
            # Calculate loss
            if binary:
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, torch.argmax(labels, dim=1))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
            
            # Calculate accuracy
            accuracy = (outputs.round() == labels).float().mean()
            total_accuracy += accuracy.item()
            total_loss += loss.item()

        avg_train_accuracy = total_accuracy / num_train_batches
        avg_train_loss = total_loss / num_train_batches

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')
        
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = len(val_dataloader)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader,total=num_val_batches):
                # Move inputs and labels to the GPU if available
                val_inputs = val_batch['image'].to(device)
                val_labels = val_batch['labels'].to(device)

                # Calculate the outputs
                val_outputs = model(val_inputs)

                if binary:
                    val_loss += criterion(val_outputs, val_labels.float()).item()
                else:
                    val_loss += criterion(val_outputs, torch.argmax(val_labels, dim=1)).item()

                val_accuracy += (val_outputs.round() == val_labels).float().mean().item()
                
                preds = torch.argmax(val_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(torch.argmax(val_labels, dim=1).cpu().numpy())

        val_loss /= num_val_batches
        val_accuracy /= num_val_batches

        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_scores.append(f1)

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, F1 Score: {f1}')

        val_losses.append(val_loss)

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()


    plt.show()

    print('Training finished.')

    # Save the trained model if needed
    if save:
        torch.save(model.state_dict(), f'Models/fine_tuned_{backbone}.pth')

    return model