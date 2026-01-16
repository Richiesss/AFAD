from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FedGenDistiller:
    """
    FedGen/DFRD方式の異種間蒸留を行うクラス
    """
    def __init__(self, generator: nn.Module, temperature: float = 4.0, device: str = 'cpu'):
        self.generator = generator
        self.temperature = temperature
        self.device = device
        self.generator.to(device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)

    def train_generator(
        self,
        ensemble_logits: torch.Tensor
    ):
        """
        Generator training step (Placeholder for specific logic if needed distinct from update)
        """
        pass

    def update(self, global_models: Dict[str, nn.Module], num_steps: int = 10):
        """
        Generatorを更新し、各ファミリーモデルへ蒸留する
        
        Args:
            global_models: 各ファミリー（または各クライアントARch）の代表モデル（重み更新済み）
        """
        self.generator.train()
        for _ in range(num_steps):
            # 1. Logit Ensemble
            z, labels = self.generator.generate_batch(32, device=self.device) # 32 is hardcoded
            logits_list = []
            
            for name, model in global_models.items():
                model.eval()
                model.to(self.device)
                with torch.no_grad():
                    out = model(z)
                    logits_list.append(out)
                    
            if not logits_list:
                continue
                
            # Average Logits
            ensemble_logits = torch.stack(logits_list).mean(dim=0)
            
            # 2. Train Generator
            # Generator loss: 
            # 「生成した画像が、Ensembleモデルによって正しく分類されるようにする」
            # これは "Class Impression" を生成することになる。
            self.optimizer.zero_grad()
            
            # Re-generate with graph for dependency
            # (Note: above 'z' generation detached? Generator.forward connects z to out?
            # self.generator.generate_batch uses internal generator forward, so 'z' (image) has grad_fn)
            # But the 'model(z)' part detached? No.
            # However, 'model' parameters are fixed (no grad). We want to update Generator.
            # So models should be fixed but allow gradient flow from input to output?
            # Standard PyTorch models allow grad w.r.t input.
            
            # Need to re-forward through generator to get graph, 
            # passed through fixed models to get loss, then backprop to generator.
            
            # But above loop used 'with torch.no_grad():' -> Blocks flow to generator.
            # Should allow grad for input z (which comes from generator).
            
            # Correct Loop:
            z_gen, labels_gen = self.generator.generate_batch(32, device=self.device)
            
            loss = 0.0
            avg_logits = 0
            for name, model in global_models.items():
                model.eval()
                # fix model params
                for p in model.parameters():
                    p.requires_grad = False
                
                out = model(z_gen)
                avg_logits += out
                
            avg_logits /= len(global_models)
            
            # Loss: CrossEntropy(avg_logits, labels)
            # Generator learns to generate images that the current ensemble classifies as 'labels'
            loss = F.cross_entropy(avg_logits, labels_gen)
            
            loss.backward()
            self.optimizer.step()
            
        self.generator.update_ema()
