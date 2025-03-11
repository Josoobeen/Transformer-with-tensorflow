from dataclasses import dataclass, field



@dataclass
class TransformerHyperParameter:
    num_heads: int = 8
    head_dims: int = 64
    encoder_block: int = 6
    decoder_block: int = 8

    vocab_size: int = 10000
    max_length: int = 1024
    

