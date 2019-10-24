import time
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, test_loader, criterion, optimizer,
                device, scheduler, train_dataset, test_dataset, num_epochs=5):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        batch_loss = 0
        batch_acc = 0
        batch = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            batch = batch + 1
            inputs = inputs.to(device)
            labels = labels.float()
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            batch_loss += loss.item() * inputs.size(0)
            final_output = torch.where(outputs > 0.5, outputs,
                                       torch.zeros(1).to(device))
            final_output = torch.where(final_output > 0.5,
                                       torch.ones(1).to(device),
                                       torch.zeros(1).to(device))

            acc = torch.sum(final_output.data == labels.data)
            running_corrects += acc.item()
            batch_acc += acc.item()/(32 * 102)

            if batch % 50 == 0:
                batch_loss = batch_loss/(32*50)
                batch_acc = batch_acc/50
                print('Batch {}, Batch Loss: {:.4f}, Acc: {:.4f}'.format(batch, batch_loss, batch_acc)) # noqa
                batch_acc = 0
                batch_loss = 0

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)

        model.eval()
        vrunning_loss = 0
        vrunning_corrects = 0
        for vinputs, vlabels in test_loader:
            vinputs = vinputs.to(device)
            vlabels = vlabels.float().to(device)
            with torch.no_grad():
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                vrunning_loss += vloss.item() * vinputs.size(0)
            vfinal_output = torch.where(voutputs > 0.5,
                                        voutputs,
                                        torch.zeros(1).to(device))
            vfinal_output = torch.where(vfinal_output > 0.5,
                                        torch.ones(1).to(device),
                                        torch.zeros(1).to(device))

            vacc = torch.sum(vfinal_output.data == vlabels.data)
            vrunning_corrects += vacc.item()/102

        vepoch_loss = vrunning_loss / len(test_dataset)
        vepoch_acc = vrunning_corrects / len(test_dataset)

        print('Train {} Epoch Loss: {:.4f} Acc: {:.4f}'.format(epoch,
                                                               epoch_loss,
                                                               epoch_acc))

        print('Validation: {} Epoch Val Loss: {:.4f} Val Acc: {:.4}'.format(epoch, vepoch_loss, vepoch_acc))# noqa

        print()

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def test_model(model, test_loader, criterion,
               device, test_dataset):
    since = time.time()
    model.eval()
    vrunning_loss = 0
    vrunning_corrects = 0
    for vinputs, vlabels in test_loader:
        vinputs = vinputs.to(device)
        vlabels = vlabels.float().to(device)
        with torch.no_grad():
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            vrunning_loss += vloss.item() * vinputs.size(0)
        vfinal_output = torch.where(voutputs > 0.5,
                                    voutputs,
                                    torch.zeros(1).to(device))
        vfinal_output = torch.where(vfinal_output > 0.5,
                                    torch.ones(1).to(device),
                                    torch.zeros(1).to(device))

        vacc = torch.sum(vfinal_output.data == vlabels.data)
        vrunning_corrects += vacc.item()/102

    vepoch_loss = vrunning_loss / len(test_dataset)
    vepoch_acc = vrunning_corrects / len(test_dataset)

    print('Test: Loss: {:.4f} Test Acc: {:.4}'.format(vepoch_loss, vepoch_acc))# noqa

    print()

    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
