import torch
import torch.nn as nn


class ObliviousDecisionTree(nn.Module):
    def __init__(self, num_features, num_classes, depth):
        super(ObliviousDecisionTree, self).__init__()
        self.num_features = num_features
        self.depth = depth
        self.num_classes = num_classes

        self.feature_weights = nn.Parameter(torch.randn(depth, num_features))
        self.thresholds = nn.Parameter(torch.randn(depth))
        self.responses = nn.Parameter(torch.randn(2 ** depth, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        decision = torch.zeros(batch_size, device=x.device)

        for d in range(self.depth):
            feature_idx = torch.argmax(self.feature_weights[d])
            feature_value = x[:, feature_idx]
            decision_bit = (feature_value > self.thresholds[d]).float()
            decision = decision * 2 + decision_bit

        return self.responses[decision.long()]


class AdvancedNODE(nn.Module):
    def __init__(self, num_trees, num_features, num_classes, depth, dropout_rate=0.3, l2_lambda=0.01):
        super(AdvancedNODE, self).__init__()
        self.trees = nn.ModuleList([ObliviousDecisionTree(num_features, num_classes, depth) for _ in range(num_trees)])
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_lambda = l2_lambda
        self.num_trees = num_trees

    def forward(self, x):
        outputs = torch.zeros(x.size(0), self.trees[0].responses.size(1), device=x.device)
        for tree in self.trees:
            tree_output = tree(x)
            outputs += tree_output
        outputs /= self.num_trees
        outputs = self.dropout(outputs)
        return outputs

    def l2_regularization(self):
        l2_loss = 0
        for tree in self.trees:
            l2_loss += torch.sum(tree.feature_weights ** 2)
            l2_loss += torch.sum(tree.thresholds ** 2)
            l2_loss += torch.sum(tree.responses ** 2)
        return self.l2_lambda * l2_loss