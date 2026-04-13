import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        # token_ids, segment_ids shape: (batch, seq_len)
        batch_size, seq_len = token_ids.shape
    
        # Lấy token embeddings theo token_ids -> (batch, seq_len, hidden_size)
        token_emb = self.token_embeddings[token_ids]
        # Lấy position embeddings theo vị trí 0..seq_len-1 -> (batch, seq_len, hidden_size)
        position_ids = np.arange(seq_len)                      # (seq_len,)
        position_emb = self.position_embeddings[position_ids]  # (seq_len, hidden_size)
    
        # Lấy segment embeddings theo segment_ids -> (batch, seq_len, hidden_size)
        segment_emb = self.segment_embeddings[segment_ids]
    
        return token_emb + position_emb + segment_emb
        pass
