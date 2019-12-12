import torch
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from torch.functional import F

import pandas as pd
import pdb
"""
Pass adjacency matrix where each entry is concatenated image features in order (col, row). 
The output should be the adjacency matrix with vector of probabilities for integer labels for relationships. 

Modeling based off https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py
"""

class GraphLabelPredict(nn.Module):
    def __init__(self, fine_tune=False):
        super(GraphLabelPredict, self).__init__()
        # Initialize feature extractor
        # self.backbone = models.resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True]) #, num_classes=10 )
        self.backbone = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers={'layer4': 'out'})
        if not fine_tune:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # FCNHead is classifier
        in_channels = 1028
        classes = 10  # actually classes
        self.pool = nn.AdaptiveAvgPool3d((1028,1,1))  # converts to shape (mxnx256x1)
        self.classifier = FCNHead(in_channels, classes)


    def forward(self, inputs, targets):
        # Simple Segmentation Model
        x = self.backbone(torch.cat(list(inputs.values())))["out"]  # outpus are [N, 2048, 32, 22]

        # Generate and adjacency matrix and prepare it for classification
        x = self._generate_graph(x)  # Becomes (N, N, 2048, 32, 22)
        x = self.pool(x)  # Becomes (N, N, 1028, 1, 1)
        x = x.reshape(1, x.shape[2], x.shape[0], x.shape[1])  # reshape to (1, 1028, N, N)
        new_shape = x.shape[-2:]
        # Predict labels for each edge connecting objects in the image
        x = self.classifier(x.cuda())  # Becomes (1, 10, N, N)
        x = F.interpolate(x, size=new_shape, mode='bilinear', align_corners=False)
        return x

    def predict_obj_relationships(self, inputs, classes):
        """
        Assumes has class predictions output from detection model. 
        Args: 
            inputs (torch.Tensor): Set each object image from predicted object detection. 
            classes (list): List of classes for each object, should length equal to the inputs.shape[0]
        """
        assert len(classes) == inputs.shape[0], "The number of classes do not match the object number."
        edge_pred = self.forward(inputs)
        edge_pred = torch.argmax(edge_pred.squeeze(), dim=0).detach().cpu().numpy()
        df = pd.DataFrame(edge_pred, index=classes, columns=classes)
        return df

    @staticmethod  
    def _generate_graph(features):
        """
        This will element-wise multiply each pair of objects and subjects. 
        """
        # edge_list = torch.combinations(torch.tensor(nodes))
        # g = nx.from_edgelist(edge_list)
        # g = torch.tensor(nx.adjacency_matrix(g).todense())
        adj = torch.ones(features.shape[0], features.shape[0], features.shape[1], features.shape[2], features.shape[3])
        for (m, n) in torch.combinations(torch.tensor(range(features.shape[0])), r=2):
            mul = torch.mul(features[m], features[n])
            self_mul = torch.mul(features[m], features[m])
            adj[m][m][:][:][:] = self_mul
            adj[m][n][:][:][:] = mul
            adj[n][m][:][:][:] = mul
        return adj