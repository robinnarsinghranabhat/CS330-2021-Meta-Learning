import argparse
import os
from datetime import datetime
from distutils.dir_util import copy_tree
from pathlib import Path

import torch
import torch.nn.functional as F
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from load_data import DataGenerator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import runtime

today_date = datetime.today().strftime("%Y-%m-%d")


class MANN(nn.Module):
    def __init__(
        self,
        num_classes,
        samples_per_class,
        model_size=128,
        input_size=784,
        model_type="LSTM",
    ):
        super(MANN, self).__init__()

        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.layer1 = torch.nn.LSTM(
            num_classes + input_size, model_size, batch_first=True
        ).float()
        self.layer2 = torch.nn.LSTM(model_size, num_classes, batch_first=True).float()
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

        self.loss_func = nn.CrossEntropyLoss()

        self.dnc = DNC(
            input_size=num_classes + input_size,
            output_size=num_classes,
            hidden_size=model_size,
            rnn_type="lstm",
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=num_classes,
            cell_size=64,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
        )

        self.model_type = model_type

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images

            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:

            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """

        batch_size, k_plus_1, num_classes, inp_dim = input_images.shape

        input_labels_cop = torch.clone(input_labels)
        input_labels_cop[:, -1, :, :] = 0
        combined_inp_and_label = torch.cat((input_images, input_labels_cop), dim=3)
        combined_inp_and_label = torch.reshape(
            combined_inp_and_label,
            (batch_size, k_plus_1 * num_classes, num_classes + inp_dim),
        )

        if "DNC" not in self.model_type:
            lstm_output_1, (h_1, c_1) = self.layer1(combined_inp_and_label)
            final_out, _ = self.layer2(lstm_output_1)

            final_out = torch.reshape(
                final_out, (batch_size, k_plus_1, num_classes, num_classes)
            )

            return final_out

        else:
            dnc_out, _ = self.dnc(combined_inp_and_label)
            dnc_out = torch.reshape(
                dnc_out, (batch_size, k_plus_1, num_classes, num_classes)
            )
            return dnc_out

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs

            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels

        Returns:
            scalar loss
        """

        # Select only the last k+1th predictions for meta-test period to backpropagate loss
        preds = preds[:, -1, :, :]
        preds = preds.permute((0, 2, 1))

        # Select only the last k+1th labels at meta-test period to backpropagate loss
        labels = labels[:, -1, :, :]
        _, labels = labels.max(dim=2)

        loss = self.loss_func(preds, labels)
        return loss


def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        model,
        optimizer,
        checkpoint["prev_loss"],
        checkpoint["epoch"],
        checkpoint["meta_test_acc"],
    )


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("USING DEVICE : ", device)

    run_name = f"K_{config.num_samples}_N_{config.num_classes}_B_{config.meta_batch_size}_H_{config.model_size}"

    # Save Artifacts : models, [ Todo : plots, summary ]
    trained_model_dir = Path(
        f"./trained_models/{today_date}/{config.model_type}/{run_name}/"
    )
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    # Save Runs : Logs, [Todo : hyperparams, ..]
    run_dir = Path(
        f"./{config.logdir}", f"{today_date}", f"{config.model_type}", f"{run_name}"
    )
    writer = SummaryWriter(run_dir)

    # Download Omniglot Dataset
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized")

    # Create Data Generator
    data_generator = DataGenerator(
        config.num_classes, config.num_samples, device=device
    )

    # Create model and optimizer
    model = (
        MANN(
            config.num_classes,
            config.num_samples,
            model_size=config.model_size,
            model_type=config.model_type,
        )
        .to(device)
        .float()
    )
    print(f"Model Initialized with : {config.model_type}")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    prev_loss = 10000  ## some high loss for first time training
    prev_step = 0  ## When was the last step upto which model was trained

    # continue from earlier trained model if available
    if (trained_model_dir / "model_ckp").exists():
        model, optim, prev_loss, prev_step, meta_test_acc = load_ckp(
            trained_model_dir / "model_ckp", model, optim
        )
        print("Continuing with last trained checkpoint ")
        print(
            f"STEP : {prev_step} || LOSS : {prev_loss} || Meta-Test Accuracy : {meta_test_acc}"
        )

    for step in range(prev_step, config.training_steps):

        images, labels = data_generator.sample_batch("train", config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch("test", config.meta_batch_size)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(
                pred,
                [-1, config.num_samples + 1, config.num_classes, config.num_classes],
            )
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)

            meta_test_acc = pred.eq(labels).double().mean().item()

            # Log to Tensorboard
            writer.add_scalar("Train Loss", train_loss, step)
            writer.add_scalar("Test Loss", test_loss, step)
            writer.add_scalar("Meta-Test Accuracy", meta_test_acc, step)

            # Help Prints and Model Saving
            test_loss = test_loss.cpu().numpy()
            train_loss = train_loss.cpu().numpy()

            print("AT STEP : ", step)
            print(" Train Loss : ", train_loss, step)
            print(" Test Loss : ", test_loss, step)

            # Saving the Model
            if test_loss < prev_loss:
                prev_loss = test_loss
                checkpoint = {
                    "epoch": step + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "prev_loss": prev_loss,
                    "meta_test_acc": meta_test_acc,
                }

                torch.save(checkpoint, trained_model_dir / f"model_ckp")
                print(f"Model Updated at : step {step}")

                ## Additionally Saving the model at Drive || For Colab
                try:
                    if config.colab_mode:
                        print("Copying Artifacts to colab")
                        save_to_colab(run_dir, trained_model_dir)
                except:
                    print("Couldn't save to Drive .. check if it's mounted")


def save_to_colab(run_dir, trained_model_dir):
    copy_tree("runs", "/content/drive/MyDrive/CS330_Artifacts/runs")
    copy_tree("trained_models", "/content/drive/MyDrive/CS330_Artifacts/trained_models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="LSTM_only")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--meta_batch_size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--training_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--model_size", type=int, default=128)
    parser.add_argument("--colab_mode", type=bool, default=0)

    main(parser.parse_args())
