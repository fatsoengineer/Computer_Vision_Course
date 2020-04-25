import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from .gradCam import GradCAM
import torch
import torchvision.transforms.functional as TF

class Visualization:
    def __init__(self, config_list):
        self.device = config_list.get("device")
        self.mean = config_list.get('mean')
        self.std = config_list.get('std')
        self.resize_shape = config_list.get('resize_shape', (32,32))
        self.model = config_list.get('model')
        self.test_loader = config_list.get('test_loader')

    def transform_to_device(self, pil_img):
        torch_img = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor()
        ])(pil_img).to(self.device)
        norm_torch_img = transforms.Normalize(self.mean, self.std)(torch_img)[None]
        return torch_img, norm_torch_img

    def visualize_cam(self, mask, img, alpha=1.0):
        """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
        Args:
            mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
            img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        Return:
            heatmap (torch.tensor): heatmap img shape of (3, H, W)
            result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
        """
        heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b]) * alpha

        result = heatmap+img.cpu()
        result = result.div(result.max()).squeeze()

        return heatmap, result

    def show_img(self, img, actual_label = None, predicted_label= None):
      #img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      fig = plt.figure(figsize=(6,6))
      plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
      plt.xlabel('Pred {} Actual {}'.format(predicted_label, actual_label))


    def plot_images(self, torch_img,normed_torch_img):
      images=[]
      g1 = GradCAM(self.model, self.model.layer1)
      g2 = GradCAM(self.model, self.model.layer2)
      g3 = GradCAM(self.model, self.model.layer3)
      g4 = GradCAM(self.model, self.model.layer4)
      mask1, _ = g1(normed_torch_img)
      mask2, _ = g2(normed_torch_img)
      mask3, _ = g3(normed_torch_img)
      mask4, _ = g4(normed_torch_img)
      heatmap1, result1 = self.visualize_cam(mask1, torch_img)
      heatmap2, result2 = self.visualize_cam(mask2, torch_img)
      heatmap3, result3 = self.visualize_cam(mask3, torch_img)
      heatmap4, result4 = self.visualize_cam(mask4, torch_img)

      images.extend([torch_img.cpu(), heatmap1, heatmap2, heatmap3, heatmap4])
      images.extend([torch_img.cpu(), result1, result2, result3, result4])
      grid_image = make_grid(images, nrow=5)
      return grid_image



    def heatmap_activations_test_loader(self, _num):
        dataiter = iter(self.test_loader)
        images, labels = dataiter.next()
        trans = transforms.ToPILImage()
        pil_img = trans(make_grid(images[_num]))
        torch_img, norm_torch_img = self.transform_to_device(pil_img)
        grid_image = self.plot_images(torch_img, norm_torch_img)
        self.show_img(grid_image)

    def heatmap_activations(self, img, actual_label = None, predicted_label= None):
        trans = transforms.ToPILImage()
        pil_img = trans(make_grid(img))
        torch_img, norm_torch_img = self.transform_to_device(pil_img)
        grid_image = self.plot_images(torch_img, norm_torch_img)
        self.show_img(grid_image, actual_label, predicted_label)        




# class UnNormalize:
#         def __init__(self, _mean, _std):
#             self.mean = _mean
#             self.std = _std
#             assert len(self.mean)== len(self.std)
#             self.unnormalize_mean =tuple([ -(mean/std) for mean, std in zip(self.mean, self.std)])
#             self.unnormalize_std = tuple([ 1/std for std in self.std])
#             self.inv_normalize = transforms.Normalize(
#                     mean=self.unnormalize_mean,
#                     std=self.unnormalize_std
#                 )

#         def __call__(self, img):
#             return self.inv_normalize(img)


class UnNormalize:
    def __init__(self, _mean, _std):
        self.mean = _mean
        self.std = _std

    def __call__(self, img):
        
        img = img*self.mean
        img = img + self.std
        img = img * 255.
        return img

