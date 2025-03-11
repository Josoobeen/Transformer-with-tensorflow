import tensorflow as tf
from tensorflow.keras.layers import Dense




class Attention(tf.keras.layers.Layer):
    """
    Multi-Head Attention class

    Attributes:
        masking_level : Multiply masking amount to which sentence needed to cover.
            - ex)
                "Hello, my name is Herion"

                first mask for train : (Hello, X, X, X, X, X)
                second mask for train : (Hello, ,, X, X, X, X)
                third mask for train : (Hello, ,, my, X, X, X)
    """
    def __init__(self, masking_level:float = -1e9):
        super().__init__()
        self.masking_level = masking_level


    def get_scaled_dot_product_attention(self, q, k, v, mask = None):
        """
        Get scaled dot product attention

        Args:
            q : quary
            k : Key
            v : Value
            mask : whether mask or not

        Explain
            - matmul q and k (self attention) q = k
            - multiply masking : Block the tokens after predict token
            - Softmax : get attention score
            - multiply with value (could be self attention(GPT, BERT) or Decoder outputs(Transformer))
            
        """
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        depth = tf.cast(tf.shape(k)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            logits += (mask * self.masking_level)
        
        attention_weights = tf.keras.layers.Softmax()(logits)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


    def call(self, q, k, v, num_heads, head_dim, mask = None):
        """
        Most importance module in "Attention is all you need".

        Args:
            q : quary
            k : Key
            v : Value
            num_heads : heads size
            head_dim : head's dim size
            mask : whether mask or not
        
        Explain
            - get same size of dementions.
            - split to head_dim and transpose : change axis for calculate attention to Embedding Layer not sequence part
                (batch_size, seq_len, 1, hidden_dim) => (batch_size, num_heads, seq_len, hidden_dim)
            - get attention score from scaled_dot_production attention
            - transpose to normal output and process
        """
        batch_size = tf.shape(q)[0]
        length = tf.shape(q)[1]
        q = Dense(num_heads * head_dim, use_bias = False)(q)
        k = Dense(num_heads * head_dim, use_bias = False)(k)
        v = Dense(num_heads * head_dim, use_bias = False)(v)

        q = tf.transpose(tf.concat(tf.split(q[:,:,tf.newaxis,:], head_dim, axis = -1), axis = 2), perm = [0, 2, 1, 3])
        k = tf.transpose(tf.concat(tf.split(k[:,:,tf.newaxis,:], head_dim, axis = -1), axis = 2), perm = [0, 2, 1, 3])
        v = tf.transpose(tf.concat(tf.split(v[:,:,tf.newaxis,:], head_dim, axis = -1), axis = 2), perm = [0, 2, 1, 3])


        scaled_attention_out, weights = self.get_scaled_dot_product_attention(q, k, v, mask = mask)

        scaled_attention_out = tf.transpose(scaled_attention_out, perm = [0,2,1,3])
        scaled_attention_out = tf.reshape(scaled_attention_out, (batch_size, length,num_heads * head_dim))
        outputs = Dense(num_heads * head_dim, use_bias = False)(scaled_attention_out)
    
        return outputs