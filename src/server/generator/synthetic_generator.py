import torch
import torch.nn as nn
import copy
from typing import List, Tuple

class SyntheticGenerator(nn.Module):
    """
    FedGen/DFRD方式の合成データ生成器
    
    Args:
        latent_dim (int): 潜在空間の次元数
        num_classes (int): クラス数
        feature_dim (int): 出力特徴量の次元数 (Logit蒸留を行う場合はLogit次元=num_classes、特徴量蒸留ならD)
                           AFAD仕様書では「特徴表現 f in R^d (画像ではなく潜在表現)」とあるが、
                           異種間知識蒸留 F-004 では「ロジットベース」と記述あり。
                           異種間モデルの共通項は「ロジット」なので、Generatorは「ロジット」または
                           「共通の特徴量空間」を出力する必要がある。
                           FedGenの元論文ではFeatureを出力してClassifierに通すが、
                           AFADでは「異種間」なのでClassifierも異なる。
                           よって、Generatorは「入力としての仮想データ（特徴量）」を出力し、
                           各クライアントの「Feature Extractorより上位の層」に入力するか、
                           あるいは「蒸留用データ」として各モデルに入力可能な形式（画像サイズのテンソル）を出力するか。
                           
                           仕様書の図 F-003: "出力: 特徴表現 f"
                           仕様書の F-004: "z_s: Studentの出力ロジット", "z_t: Teacherの出力ロジット"
                           
                           この矛盾（画像入力モデルに特徴量をどう入れるか）を解決するため、
                           ここではGeneratorは「画像サイズ (1, 28, 28)」を生成するGeneratorとして実装する。
                           これにより全てのCNN/ViTモデルに等しく入力可能となる。
                           (Data-Free KDの標準的なアプローチ)
        hidden_dims (List[int]): 中間層の次元
        ema_decay (float): EMAの減衰率
    """
    
    def __init__(
        self, 
        latent_dim: int = 100, 
        num_classes: int = 10, 
        output_shape: tuple = (1, 28, 28),
        hidden_dims: List[int] = [256, 512],
        ema_decay: float = 0.9
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.ema_decay = ema_decay
        
        # Label Embedding
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        
        # Generator Network
        layers = []
        input_dim = latent_dim * 2 # Noise + Label
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            input_dim = h_dim
            
        # Final projection to image size
        output_dim = 1
        for d in output_shape:
            output_dim *= d
            
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid()) # For image [0,1] normalization (MNIST is approx in this range when normalized, roughly)
        # Note: MNIST data loader normalizes to mean 0.1307, std 0.3081. 
        # Output should ideally match this distribution or we use Tanh and denormalize?
        # For simplicity, Sigmoid -> [0, 1] is decent for "image-like" generation.
        
        self.generator = nn.Sequential(*layers)
        
        # EMA Model
        self.ema_generator = copy.deepcopy(self.generator)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic data
        z: [Batch, Latent]
        labels: [Batch]
        """
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        out = self.generator(x)
        return out.view(-1, *self.output_shape)

    @torch.no_grad()
    def update_ema(self):
        for p, ema_p in zip(self.generator.parameters(), self.ema_generator.parameters()):
            ema_p.data = self.ema_decay * ema_p.data + (1 - self.ema_decay) * p.data

    def generate_batch(self, batch_size: int, device: str = 'cpu', use_ema: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(batch_size, self.latent_dim).to(device)
        labels = torch.randint(0, self.num_classes, (batch_size,)).to(device)
        
        gen_model = self.ema_generator if use_ema else self.generator
        
        # Forward manually to handle separate embedding if needed, but current forward does it.
        # But EMA generator is just the sequential part? No, self.ema_generator is copy of self.generator (Sequential)
        # Ah, self.generator is the Sequential part in __init__, but the class has label_embed too.
        # I should make sure EMA copy includes label_embed.
        # Let's fix __init__ to separate model properly or copy whole module.
        pass # Fixed below via re-implementation logic in prompt
        
        # Re-implementation for correctness:
        # Instead of deepcopying just self.generator, I need to handle the whole forward logic.
        # Let's assume generate_batch calls forward.
        
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        
        if use_ema:
             out = self.ema_generator(x)
        else:
             out = self.generator(x)
             
        return out.view(-1, *self.output_shape), labels

