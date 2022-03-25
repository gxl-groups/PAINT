import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from core.utils import set_device
import torchvision
from typing import List

'''
  most codes are borrowed from: 
  https://github.com/knazeri/edge-connect/blob/master/src/loss.py
'''




class VGG19(torch.nn.Module):
  def __init__(self):
    super(VGG19, self).__init__()
    features = models.vgg19(pretrained=True).features
    self.relu1_1 = torch.nn.Sequential()
    self.relu1_2 = torch.nn.Sequential()

    self.relu2_1 = torch.nn.Sequential()
    self.relu2_2 = torch.nn.Sequential()

    self.relu3_1 = torch.nn.Sequential()
    self.relu3_2 = torch.nn.Sequential()
    self.relu3_3 = torch.nn.Sequential()
    self.relu3_4 = torch.nn.Sequential()

    self.relu4_1 = torch.nn.Sequential()
    self.relu4_2 = torch.nn.Sequential()
    self.relu4_3 = torch.nn.Sequential()
    self.relu4_4 = torch.nn.Sequential()

    self.relu5_1 = torch.nn.Sequential()
    self.relu5_2 = torch.nn.Sequential()
    self.relu5_3 = torch.nn.Sequential()
    self.relu5_4 = torch.nn.Sequential()

    for x in range(2):
      self.relu1_1.add_module(str(x), features[x])

    for x in range(2, 4):
      self.relu1_2.add_module(str(x), features[x])

    for x in range(4, 7):
      self.relu2_1.add_module(str(x), features[x])

    for x in range(7, 9):
      self.relu2_2.add_module(str(x), features[x])

    for x in range(9, 12):
      self.relu3_1.add_module(str(x), features[x])

    for x in range(12, 14):
      self.relu3_2.add_module(str(x), features[x])

    for x in range(14, 16):
      self.relu3_2.add_module(str(x), features[x])

    for x in range(16, 18):
      self.relu3_4.add_module(str(x), features[x])

    for x in range(18, 21):
      self.relu4_1.add_module(str(x), features[x])

    for x in range(21, 23):
      self.relu4_2.add_module(str(x), features[x])

    for x in range(23, 25):
      self.relu4_3.add_module(str(x), features[x])

    for x in range(25, 27):
      self.relu4_4.add_module(str(x), features[x])

    for x in range(27, 30):
      self.relu5_1.add_module(str(x), features[x])

    for x in range(30, 32):
      self.relu5_2.add_module(str(x), features[x])

    for x in range(32, 34):
      self.relu5_3.add_module(str(x), features[x])

    for x in range(34, 36):
      self.relu5_4.add_module(str(x), features[x])

    # don't need the gradients, just want the features
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    relu1_1 = self.relu1_1(x)
    relu1_2 = self.relu1_2(relu1_1)

    relu2_1 = self.relu2_1(relu1_2)
    relu2_2 = self.relu2_2(relu2_1)

    relu3_1 = self.relu3_1(relu2_2)
    relu3_2 = self.relu3_2(relu3_1)
    relu3_3 = self.relu3_3(relu3_2)
    relu3_4 = self.relu3_4(relu3_3)

    relu4_1 = self.relu4_1(relu3_4)
    relu4_2 = self.relu4_2(relu4_1)
    relu4_3 = self.relu4_3(relu4_2)
    relu4_4 = self.relu4_4(relu4_3)

    relu5_1 = self.relu5_1(relu4_4)
    relu5_2 = self.relu5_2(relu5_1)
    relu5_3 = self.relu5_3(relu5_2)
    relu5_4 = self.relu5_4(relu5_3)

    out = {
      'relu1_1': relu1_1,
      'relu1_2': relu1_2,

      'relu2_1': relu2_1,
      'relu2_2': relu2_2,

      'relu3_1': relu3_1,
      'relu3_2': relu3_2,
      'relu3_3': relu3_3,
      'relu3_4': relu3_4,

      'relu4_1': relu4_1,
      'relu4_2': relu4_2,
      'relu4_3': relu4_3,
      'relu4_4': relu4_4,

      'relu5_1': relu5_1,
      'relu5_2': relu5_2,
      'relu5_3': relu5_3,
      'relu5_4': relu5_4,
    }
    return out


class SemanticReconstructionLoss(nn.Module):
  '''
  Implementation of the proposed semantic reconstruction loss
  '''

  def __init__(self, weight_factor: float = 0.1) -> None:
    '''
    Constructor
    '''
    # Call super constructor
    super(SemanticReconstructionLoss, self).__init__()
    # Save parameter
    self.weight_factor = weight_factor
    # Init max pooling operations. Since the features have various dimensions, 2d & 1d max pool as the be init
    self.max_pooling_2d = nn.MaxPool2d(2)
    self.max_pooling_1d = nn.MaxPool1d(2)
    self.vgg = VGG16().cuda()
    self.vgg.eval()

  def __repr__(self):
    '''
    Get representation of the loss module
    :return: (str) String including information
    '''
    return '{}, weights factor={}, maxpool kernel size{}' \
      .format(self.__class__.__name__, self.weight_factor, self.max_pooling_1d.kernel_size)

  def forward(self, features_real: List[torch.Tensor], features_fake: List[torch.Tensor]) -> torch.Tensor:
    '''
    Forward pass
    :param features_real: (List[torch.Tensor]) List of real features
    :param features_fake: (List[torch.Tensor]) List of fake features
    :return: (torch.Tensor) Loss
    '''
    # Check lengths
    assert len(features_real) == len(features_fake)
    # Init loss
    loss = torch.tensor(0.0, dtype=torch.float32, device=features_real[0].device)
    # Calc full loss
    features_real = self.vgg(features_real)
    features_fake = self.vgg(features_fake)
    for feature_real, feature_fake in zip(features_real, features_fake):
      # Downscale features
      if len(feature_fake.shape) == 4:
        feature_real = self.max_pooling_2d(feature_real)
        feature_fake = self.max_pooling_2d(feature_fake)
      else:
        feature_real = self.max_pooling_1d(feature_real.unsqueeze(dim=1))
        feature_fake = self.max_pooling_1d(feature_fake.unsqueeze(dim=1))
      # Normalize features
      union = torch.cat((feature_real, feature_fake), dim=0)
      feature_real = (feature_real - union.mean()) / union.std()
      feature_fake = (feature_fake - union.mean()) / union.std()
      # Calc l1 loss of the real and fake feature conditionalized by the corresponding mask
      loss = loss + torch.mean(torch.abs((feature_real - feature_fake)))
    # Average loss with number of features
    loss = loss / len(features_real)
    return self.weight_factor * loss


class VGG16(nn.Module):
  '''
  Implementation of a pre-trained VGG 16 model which outputs intermediate feature activations of the model.
  '''

  def __init__(self, path_to_pre_trained_model: str = None) -> None:
    '''
    Constructor
    :param pretrained: (bool) True if the default pre trained vgg16 model pre trained in image net should be used
    '''
    # Call super constructor
    super(VGG16, self).__init__()
    # Load model
    if path_to_pre_trained_model is not None:
      self.vgg16 = torch.load(path_to_pre_trained_model)
    else:
      self.vgg16 = torchvision.models.vgg16(pretrained=False)
    # Convert feature module into model list
    self.vgg16.features = nn.ModuleList(list(self.vgg16.features))
    # Convert classifier into module list
    self.vgg16.classifier = nn.ModuleList(list(self.vgg16.classifier))

  def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
    '''
    Forward pass of the model
    :param input: (torch.Tenor) Input tensor of shape (batch size, channels, height, width)
    :return: (List[torch.Tensor]) List of intermediate features in ascending oder w.r.t. the number VGG layer
    '''
    # Adopt grayscale to rgb if needed
    if input.shape[1] == 1:
      output = input.repeat_interleave(3, dim=1)
    else:
      output = input
    # Init list for features
    features = []
    # Feature path
    for layer in self.vgg16.features:
      output = layer(output)
      if isinstance(layer, nn.MaxPool2d):
        features.append(output)
    # Average pool operation
    output = self.vgg16.avgpool(output)
    # Flatten tensor
    output = output.flatten(start_dim=1)
    # Classification path
    for index, layer in enumerate(self.vgg16.classifier):
      output = layer(output)
      if index == 3 or index == 6:
        features.append(output)
    return features


class AdversarialLoss(nn.Module):
  r"""
  Adversarial loss
  https://arxiv.org/abs/1711.10337
  """

  def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
    r"""
    type = nsgan | lsgan | hinge
    """
    super(AdversarialLoss, self).__init__()
    self.type = type
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))

    if type == 'nsgan':
      self.criterion = nn.BCELoss()
    elif type == 'lsgan':
      self.criterion = nn.MSELoss()
    elif type == 'hinge':
      self.criterion = nn.ReLU()

  def patchgan(self, outputs, is_real=None, is_disc=None):
    if self.type == 'hinge':
      if is_disc:
        if is_real:
          outputs = -outputs
        return self.criterion(1 + outputs).mean()
      else:
        return (-outputs).mean()
    else:
      labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
      loss = self.criterion(outputs, labels)
      return loss

  def __call__(self, outputs, is_real=None, is_disc=None):
    return self.patchgan(outputs, is_real, is_disc)


class PerceptualLoss(nn.Module):
  r"""
  Perceptual loss, VGG-based
  https://arxiv.org/abs/1603.08155
  https://github.com/dxyang/StyleTransfer/blob/master/utils.py
  """

  def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
    super(PerceptualLoss, self).__init__()
    self.add_module('vgg', VGG19())
    self.criterion = torch.nn.L1Loss()
    self.weights = weights

  def __call__(self, x, y):
    # Compute features
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)
    content_loss = 0.0
    content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
    content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
    content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
    content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
    content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

    return content_loss


class StyleLoss(nn.Module):
  r"""
  Perceptual loss, VGG-based
  https://arxiv.org/abs/1603.08155
  https://github.com/dxyang/StyleTransfer/blob/master/utils.py
  """

  def __init__(self):
    super(StyleLoss, self).__init__()
    self.add_module('vgg', VGG19())
    self.criterion = torch.nn.L1Loss()

  def compute_gram(self, x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)

    return G

  def __call__(self, x, y):
    # Compute features
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)

    # Compute loss
    style_loss = 0.0
    style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

    return style_loss


class TVLoss(nn.Module):
  def __init__(self, TVLoss_weight=3):
    super(TVLoss, self).__init__()
    self.TVLoss_weight = TVLoss_weight

  def forward(self, x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = self._tensor_size(x[:, :, 1:, :])
    count_w = self._tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

  def _tensor_size(self, t):
    return t.size()[1] * t.size()[2] * t.size()[3]
