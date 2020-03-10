'''
Training class to train, test and visualize
TODO: Visualize modules
'''
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF # Required for misclassified images.

import matplotlib.pyplot as plt

# %matplotlib inline
SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)


class Train:

    def __init__(self, configs):
        self.device = configs.get("device")
        self.model = configs.get("model")
        self.EPOCHS = configs.get("EPOCHS")
        self.criterion = configs.get("criterion")
        self.optimizer = configs.get("optimizer")
        self.scheduler = configs.get("scheduler", None)
        self.logger = configs.get("logger")
        self.train_loader = configs.get("train_loader")
        self.test_loader = configs.get("test_loader")
        self.flag_misclassified_images = configs.get("flag_misclassified_images", False)
        

        self.final_train_loss = 0
        self.final_train_acc = 0
        self.final_test_loss = 0
        self.final_test_acc  = 0
        self.epochs_ran = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_losses_list = []
        self.test_acc_list = []
        self.misclassified_images_list = []

    def plot_loss_graph(self):
        plt.figure(figsize=[10,6])
        plt.plot(self.train_loss_list)
        plt.plot(self.test_losses_list)
        plt.legend(['Train Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"{self.__class__.__name__} Model Loss")
        plt.savefig(f"{self.__class__.__name__} Model Loss.png")
    
    def plot_acc_graph(self):
        plt.figure(figsize=[10,6])
        plt.plot(self.train_acc_list)
        plt.plot(self.test_acc_list)
        plt.legend(['Train Acc', 'Test Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.title(f"{self.__class__.__name__} Model Accuracy")
        plt.savefig(f"{self.__class__.__name__} Model Accuracy.png")

    def train_model(self):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        train_loss = 0
        train_acc = 0

        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(self.device), target.to(self.device)

            # Init
            self.optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            loss = self.criterion(y_pred, target)
            train_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()


            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        train_acc = 100*correct/processed
        train_loss /= len(self.train_loader.dataset)
        return train_loss, train_acc


    def test_model(self):
        misclassified_images_list = []
        self.model.eval()
        test_loss = 0
        test_acc = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                
                #TODO: To be fixed to auto - normalize  and only do this on last epoch and modularize
                if self.flag_misclassified_images:
                    # Check images
                    wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
                    wrong_samples = data[wrong_idx]
                    pred_label_list = pred[wrong_idx]
                    actual_label_list = target.view_as(pred)[wrong_idx]

                    for idx, img in enumerate(wrong_samples):
                        img = img.cpu()
                        pred_label = pred_label_list[idx].cpu()
                        actual_label = actual_label_list[idx].cpu()

                        # Undo normalization
                        img = img * 0.3081
                        img = img + 0.1307
                        img = img * 255.
                        img = img.byte()
                        img = TF.to_pil_image(img)
                        misclassified_images_list.append({
                            "image": img,
                            "actual_label": actual_label,
                            "predicted_label": pred_label,
                        })

        test_loss /= len(self.test_loader.dataset)
        test_acc = 100. * correct / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            test_acc))
        
        return test_loss, test_acc, misclassified_images_list


    def run_model(self):

        for epoch in range(1,self.EPOCHS+1):
            print(f"EPOCH:{epoch} | LR: {self.scheduler.get_lr()}")

            train_loss, train_acc= self.train_model()
            test_loss, test_acc, self.misclassified_images_list= self.test_model()

            self.final_train_loss = train_loss
            self.final_train_acc = train_acc
            self.final_test_loss = test_loss
            self.final_test_acc  = test_acc
            self.epochs_ran = epoch


            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.test_losses_list.append(test_loss)
            self.test_acc_list.append(test_acc)

            if self.logger is not None:
                pass
                # TODO  self.logger.write([self.vals[0], self.vals[1], epoch, self.optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc])
        
        # return self.train_loss_list, self.train_acc_list, self.test_losses_list, self.test_acc_list, self.misclassified_images_list


