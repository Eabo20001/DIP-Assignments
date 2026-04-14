import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR


def build_parser():
    parser = argparse.ArgumentParser(description='Train the FCN pix2pix baseline.')
    parser.add_argument('--train-list', default='train_list.txt', help='path to the training image list')
    parser.add_argument('--val-list', default='val_list.txt', help='path to the validation image list')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--image-size', type=int, default=256, help='square size used for resizing paired images')
    parser.add_argument('--num-workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--scheduler-step-size', type=int, default=200, help='step size for LR decay')
    parser.add_argument('--scheduler-gamma', type=float, default=0.2, help='LR decay factor')
    parser.add_argument('--save-every', type=int, default=50, help='checkpoint frequency in epochs')
    parser.add_argument('--sample-every', type=int, default=5, help='visualization frequency in epochs')
    parser.add_argument('--sample-count', type=int, default=5, help='number of images to save per visualization')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='directory for checkpoints')
    parser.add_argument('--train-results-dir', default='train_results', help='directory for training samples')
    parser.add_argument('--val-results-dir', default='val_results', help='directory for validation samples')
    return parser

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    num_images = min(num_images, inputs.size(0))
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    num_epochs,
    sample_every,
    sample_count,
    train_results_dir,
):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_rgb)

        # Save sample images every 5 epochs
        if epoch % sample_every == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, train_results_dir, epoch, sample_count)

        # Compute the loss
        loss = criterion(outputs, image_semantic)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs, sample_every, sample_count, val_results_dir):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % sample_every == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, val_results_dir, epoch, sample_count)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    args = build_parser().parse_args()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file=args.train_list, image_size=args.image_size)
    val_dataset = FacadesDataset(list_file=args.val_list, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # Training loop
    num_epochs = args.epochs
    print(f'Using device: {device}')
    print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    for epoch in range(num_epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
            args.sample_every,
            args.sample_count,
            args.train_results_dir,
        )
        validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            num_epochs,
            args.sample_every,
            args.sample_count,
            args.val_results_dir,
        )

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 50 epochs
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.checkpoint_dir}/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
