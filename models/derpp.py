import statistics

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from utils.buffer import Buffer
from utils.metrics import backward_transfer, forgetting, forward_transfer
from utils.evaluation import evaluate_past, test_epoch, evaluate_next
from utils.utils import binary_accuracy

def train_derpp(train_set, test_set, model, loss, optimizer, device, config, suffix):
    """
    :param train_set: Train set
    :param test_set: Test set
    :param model: PyTorch model
    :param loss: loss function
    :param optimizer: optimizer
    :param device: device cuda/cpu
    :param config: configuration
    :param suffix: Suffix for the filename and Summary Writer
    """

    wandb.init(project="LOD2022", entity="albertozurli", reinit=True)
    # train_writer = SummaryWriter('./runs/dark_exp_replay/train/' + suffix)
    derpp = Derpp(config, device, model, loss, optimizer)
    accuracy = []

    # N SummaryWriter for N domains
    if config['evaluate']:
        text_file = open("derpp" + suffix + ".txt", "a")
        text_file.write("\nCONTINUAL LEARNING W\\ DER++ \n")
        # test_writer = SummaryWriter('./runs/derpp/test/' + suffix)
        # writer_list = []
        test_list = [[] for _ in range(len(train_set))]
        # for i in range(len(train_set)):
        #     writer_list.append(SummaryWriter(f'./runs/derpp/test/{suffix}/d_{i}'))

    # Eval without training
    _, _, random_mean_accuracy, _ = evaluate_past(derpp.model, len(test_set) - 1, test_set, loss, device)

    for index, data_set in enumerate(train_set):
        derpp.model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            derpp.model.train()
            epoch_loss = []
            epoch_acc = []
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = x.to(device)
                labels = y.to(device)
                output = derpp.model(inputs)

                first_loss = loss(output, labels.squeeze(1))
                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred, labels.squeeze(1))

                if not derpp.buffer.is_empty():
                    buf_input, _, buf_logit = derpp.buffer.get_data(config['batch_size'])
                    buf_input = torch.stack(buf_input)
                    buf_logit = torch.stack(buf_logit)
                    buf_output = derpp.model(buf_input)
                    add_loss = F.mse_loss(buf_output, buf_logit)
                    final_loss = first_loss + config['alpha'] * add_loss.data

                    buf_input, buf_label, _ = derpp.buffer.get_data(config['batch_size'])
                    buf_input = torch.stack(buf_input)
                    buf_label = torch.stack(buf_label)
                    buf_output = derpp.model(buf_input)
                    final_loss += config['beta'] * loss(buf_output,buf_label.squeeze(1))
                else:
                    final_loss = first_loss

                if config['cnn']:
                    l1_reg = 0
                    for param in derpp.model.parameters():
                        l1_reg += torch.norm(param, 1)
                    final_loss += config['l1_lambda'] * l1_reg

                epoch_loss.append(final_loss.item())
                epoch_acc.append(acc.item())

                final_loss.backward()
                optimizer.step()

                if epoch == 0:
                    derpp.buffer.add_data(examples=x.to(device), labels=labels,logits=output.to(device))

            wandb.log({"Train/loss": statistics.mean(epoch_loss),
                       "Train/accuracy": statistics.mean(epoch_acc)})
            # train_writer.add_scalar('Train/loss', statistics.mean(epoch_loss),
            #                         epoch + (config['epochs'] * index))
            # train_writer.add_scalar('Train/accuracy', statistics.mean(epoch_acc),
            #                         epoch + (config['epochs'] * index))

            if (epoch % 100 == 0) or (epoch == config['epochs'] - 1):
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Test each epoch
            if config['evaluate']:
                tmp_list = []
                # Past tasks
                for past in range(index):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp, _ = test_epoch(derpp.model, test_loader, loss, device)
                    wandb.log({f"Test/domain{past}_acc":statistics.mean(tmp)})
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(derpp.model, test_loader, loss, device)
                wandb.log({f"Test/domain{index}_acc": statistics.mean(tmp),
                           "Test/domain_loss": statistics.mean(loss_task)})
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)
                avg = sum(tmp_list) / len(tmp_list)
                wandb.log({"Test/mean_acc": avg})

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(derpp.model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(derpp.model, index, test_set, loss, device))

        torch.save(derpp.model.state_dict(), f'checkpoints/der/model_d{index}.pt')

    # Check buffer distribution
    derpp.buffer.check_distribution()

    # Compute transfer metrics
    backward = backward_transfer(accuracy)
    forward = forward_transfer(accuracy, random_mean_accuracy)
    forget = forgetting(accuracy)
    print(f'\nBackward transfer: {backward}')
    print(f'Forward transfer: {forward}')
    print(f'Forgetting: {forget}')

    if config['evaluate']:
        text_file.write(f"Backward: {backward}\n")
        text_file.write(f"Forward: {forward}\n")
        text_file.write(f"Forgetting: {forget}\n")
        text_file.close()

        df = pd.DataFrame(test_list)
        df.to_csv(f'derpp_{suffix}.csv')


class Derpp:
    def __init__(self, config, device, model, loss, optimizer):
        self.config = config
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.buffer = Buffer(self.config['buffer_size'], self.device)