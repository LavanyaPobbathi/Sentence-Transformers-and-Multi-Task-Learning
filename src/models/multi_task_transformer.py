import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional, Union, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformer(nn.Module):
    """
    A sentence transformer that converts text into fixed-length embeddings.
    
    Architectural Choices:
    1. Pooling Strategy: 
       - Implements both mean pooling and CLS token pooling
       - Mean pooling chosen as default since it better captures sentence semantics
       - CLS token pooling available for BERT-style classification tasks
    
    2. Embedding Normalization:
       - Optional L2 normalization for similarity tasks
       - Important for cosine similarity and semantic search applications
    
    3. Flexible Dimensionality:
       - Optional projection layer to adjust embedding dimension
       - Allows compatibility with different downstream tasks
    
    4. Attention Masking:
       - Proper handling of variable-length inputs
       - Ensures padding tokens don't affect sentence representations
    """
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        pooling_strategy: str = "mean",
        normalize_embeddings: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        
        # Add projection layer if needed
        self.projection = None
        if embedding_dim != self.transformer.config.hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.Dropout(dropout_rate),
                nn.GELU()
            )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to generate sentence embeddings.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention masks
            
        Returns:
            torch.Tensor: Sentence embeddings
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool token embeddings based on strategy
        if self.pooling_strategy == "mean":
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:  # CLS token pooling
            embeddings = outputs.last_hidden_state[:, 0]
        
        if self.projection is not None:
            embeddings = self.projection(embeddings)
            
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings


class MultiTaskTransformer(nn.Module):
    """
    Multi-task transformer supporting sentence classification and auxiliary NLP tasks.
    
    Architecture Design Choices:
    1. Shared Backbone:
       - Single transformer for feature extraction
       - Enables knowledge sharing between tasks
       - Reduces total parameter count
    
    2. Task-Specific Heads:
       - Separate networks for each task
       - Different architectures based on task requirements
       - Enhanced with layer normalization and residual connections
    
    3. Transfer Learning Support:
       - Optional backbone freezing
       - Useful for low-resource scenarios
    
    4. Flexible Task Selection:
       - Can run single or multiple tasks
       - Task-specific or joint inference
    """
    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        num_classes_a: int = 3,
        num_classes_b: int = 2,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Initialize backbone
        self.backbone = SentenceTransformer(
            model_name=base_model,
            normalize_embeddings=False  # No normalization for classification
        )
        
        if freeze_backbone:
            logger.info("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        hidden_size = self.backbone.transformer.config.hidden_size
        
        # Enhanced Task A head (Sentiment Analysis)
        self.task_a_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes_a)
        )
        
        # Enhanced Task B head (Subjectivity Classification)
        self.task_b_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes_b)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for multi-task inference.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            task: Specific task to run ('a', 'b', or None for both)
            
        Returns:
            Single task output or tuple of task outputs
        """
        embeddings = self.backbone(input_ids, attention_mask)
        
        if task == "a":
            return self.task_a_head(embeddings)
        elif task == "b":
            return self.task_b_head(embeddings)
        else:
            return (
                self.task_a_head(embeddings),
                self.task_b_head(embeddings)
            )