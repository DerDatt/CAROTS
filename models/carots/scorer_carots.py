import copy

import torch
import torch.nn.functional as F


class Scorer():
    def __init__(self, cfg, model):
        super(Scorer, self).__init__()
        self.cfg = cfg
        self.model = model

        self.centroid_wo_norm, self.centroid = self.init_centroid()
    
    @torch.no_grad()
    def init_centroid(self):
        from datasets.loader import construct_loader
        from trainer import prepare_inputs

        is_training = copy.deepcopy(self.model.training)
        self.model.eval()
        
        loader = construct_loader(self.cfg, split="train")
        outputs = []
        for inputs in loader:
            inputs, _ = prepare_inputs(inputs)
            if self.cfg.CAROTS.POSITIVE_AUGMENTOR.ENABLE:
                output = self.model(inputs, positive_augment=True, negative_augment=True)
            else:
                output = self.model(inputs, positive_augment=False, negative_augment=True)
            outputs.append(output[:(len(output) // 2)])
            
        outputs = torch.concat(outputs, dim=0)
        centroid_wo_norm = torch.mean(outputs, dim=0, keepdim=True)
        
        outputs = F.normalize(outputs, p=2, dim=1)
        centroid = torch.mean(outputs, dim=0, keepdim=True)

        if is_training:
            self.model.train()

        return centroid_wo_norm, centroid
    
    @torch.no_grad()
    def get_anomaly_scores(self, x):
        is_training = copy.deepcopy(self.model.training)
        self.model.eval()

        if self.cfg.SCORER.TYPE == "causal_discoverer":            
            x, y = x[:, :self.cfg.CUTS_PLUS.INPUT_STEP], x[:, self.cfg.CUTS_PLUS.INPUT_STEP:]
            Graph = (self.model.causal_discoverer.causality_mtx > 0.5).float()
            Graph = Graph[None].expand(x.size(0), -1, -1)
            y_pred = self.model.causal_discoverer(x, Graph)
            y_pred = y_pred.transpose(1, 2)
            assert y.shape == y_pred.shape  #  (B, T, N)
            score = F.mse_loss(y_pred, y, reduction='none').mean(dim=(1, 2))
            return score
        
        x = self.model(x, positive_augment=False, negative_augment=False)

        if self.cfg.SCORER.TYPE == "l2":
            score = torch.cdist(x, self.centroid_wo_norm).squeeze()
        elif self.cfg.SCORER.TYPE == "cos":
            centroid_normalized = F.normalize(self.centroid, dim=1)
            x_normalized = F.normalize(x, dim=1)
            score = -torch.matmul(x_normalized, centroid_normalized.t()).squeeze() * 0.5 + 0.5
        else:
            raise ValueError("Unsupported SCORER.TYPE")

        if is_training:
            self.model.train()

        return score

    @torch.no_grad()
    def get_per_variable_cd_scores(self, x):
        """Per-variable forecasting error for variable-level anomaly localization.

        This is the building block for Step 2 (variable-level localization). It
        reuses the *exact same* causal-discoverer forecast that produces the
        causal-discrepancy (CD) anomaly score in ``get_anomaly_scores`` above,
        but it stops one step earlier: instead of averaging the squared error
        over *both* time and variables to obtain a single scalar per window, it
        averages over time only and keeps the variable axis intact.

        Args:
            x: A batch of windows with shape ``(B, WIN_SIZE, N)``.

        Returns:
            A tensor of shape ``(B, N)`` whose entry ``[b, n]`` is the mean
            squared forecasting error of variable ``n`` in window ``b``. A large
            value means the causal discoverer could not predict that variable
            well from its parents, i.e. that variable looks anomalous.

        Notes:
            This method is inference-only. It does not touch training, the loss
            functions or the model weights, exactly as required for Step 2.
        """
        is_training = copy.deepcopy(self.model.training)
        self.model.eval()

        # Split each window into the input portion (fed to the causal
        # discoverer) and the target portion (what it must forecast). This is
        # identical to the "causal_discoverer" branch of get_anomaly_scores.
        x, y = x[:, :self.cfg.CUTS_PLUS.INPUT_STEP], x[:, self.cfg.CUTS_PLUS.INPUT_STEP:]
        Graph = (self.model.causal_discoverer.causality_mtx > 0.5).float()
        Graph = Graph[None].expand(x.size(0), -1, -1)
        y_pred = self.model.causal_discoverer(x, Graph)
        y_pred = y_pred.transpose(1, 2)
        assert y.shape == y_pred.shape  # (B, T, N)

        # Mean over the time axis only -> one error value per variable.
        score = F.mse_loss(y_pred, y, reduction='none').mean(dim=1)  # (B, N)

        if is_training:
            self.model.train()

        return score

