import torch
import torch.nn as nn


class FeatExtractWrapper(torch.nn.Module):
    def __init__(self, feat_extractor, classifier):
        super(FeatExtractWrapper, self).__init__()
        self.feat_extractor = feat_extractor
        self.classifier = classifier

    def forward(self, x):
        feat = self.feat_extractor(x)
        x = self.classifier(feat)
        return feat, x


class FeatExtractor(nn.Module):
    """
    extract feature from a specific layer.
    model should have a forward method that returns a dict of features.
    """

    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name

    def forward(self, x):
        feat = self.model(x, get_feat=True)[1][self.layer_name]
        if len(feat.shape) == 4:
            feat = F.avg_pool2d(feat, feat.shape[2:])
            feat = feat.view(feat.shape[0], -1)
        return feat


class NormWrapper(torch.nn.Module):
    def __init__(self, net, norm=None):
        super(NormWrapper, self).__init__()
        if norm is None:

            def norm(x):
                return x * 2 - 1

        self.norm = norm
        self.net = net

    def forward(self, *x, **kwargs):
        get_feat = kwargs.get("get_feat", False)

        if len(x) == 2:
            x = [self.norm(x_) for x_ in x]
            if get_feat:
                x, feat = self.net(x[0], x[1], **kwargs)
            else:
                x = self.net(x[0], x[1], **kwargs)
        else:
            if isinstance(x, tuple):
                x = x[0]
            x = self.norm(x)
            if get_feat:
                x, feat = self.net(x, **kwargs)
            else:
                x = self.net(x, **kwargs)
        if get_feat:
            return x, feat
        else:
            return x


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor(std).reshape(1, 3, 1, 1).cuda()

    def __call__(self, x):
        return (x - self.mean) / self.std


if __name__ == "__main__":
    norm = Normalizer([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    x = torch.randn(20, 3, 32, 32).cuda()
    x = norm(x)
