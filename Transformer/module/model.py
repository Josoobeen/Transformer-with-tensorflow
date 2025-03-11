from module.mask import create_padding_mask, create_look_ahead_mask
from module.positional_encoding import positional_encoding
from module.sublayer_connection import sublayer_connection
from HyperParameter.transformer import TransformerHyperParameter
from module.attention import Attention

import tensorflow as tf





class Transformer:
    def __init__(
            self,
            hyperparameter: TransformerHyperParameter
    ):
        self.hp = hyperparameter
        self.attention = Attention(masking_level=-1e9)
        self.look_ahead_mask = create_look_ahead_mask(self.hp.max_length)
    
    def get_model(self) -> tf.keras.models.Model:
        """
        Build Transformer Model
        """
        encoder_in = tf.keras.layers.Input(shape = (self.hp.max_length,))
        encoder_embedding = tf.keras.layers.Embedding(
            input_dim = self.hp.vocab_size, 
            output_dim = self.hp.num_heads * self.hp.head_dims
        )(encoder_in)

        # Encoder positional Encoding part
        position = positional_encoding(
            self.hp.num_heads * self.hp.head_dims,
            self.hp.max_length
        )
        position = tf.expand_dims(position, 0)

        encoder_inputs = tf.keras.layers.Add()([encoder_embedding, position])

        # encoder block
        encoder_attention = self.build_encoder_block(encoder_inputs)



        decoder_in = tf.keras.layers.Input(shape = (self.hp.max_length,))
        decoder_embedding = tf.keras.layers.Embedding(
            input_dim = self.hp.vocab_size, 
            output_dim = self.hp.num_heads * self.hp.head_dims
        )(decoder_in)
        
        # Decoder positional Encoding part
        decoder_inputs = tf.keras.layers.Add()([decoder_embedding, position])

        # encoder block
        decoder_attention = self.build_decoder_block(encoder_attention, decoder_inputs, mask = self.look_ahead_mask)

        # Linear output
        outputs = tf.keras.layers.Dense(self.hp.vocab_size)(decoder_attention)
        outputs = tf.keras.activations.softmax(outputs)

        model = tf.keras.models.Model(inputs = [encoder_inputs, decoder_in], outputs = outputs)

        return model


    def build_encoder_block(self, encoder_inputs):
        for _ in range(self.hp.encoder_block):
            multihead_attention = self.attention(
                q = encoder_inputs,
                k = encoder_inputs,
                v = encoder_inputs,
                num_heads = self.hp.num_heads,
                head_dim = self.hp.head_dims,
                mask = None,
            )


            encoder_inputs = sublayer_connection()(encoder_inputs, multihead_attention)

            ff1 = tf.keras.layers.Dense(self.hp.num_heads * self.hp.head_dims * 4)(encoder_inputs)
            ff1_d = tf.keras.layers.Dropout(0.2)(ff1)
            ff2 = tf.keras.layers.Dense(self.hp.num_heads * self.hp.head_dims)(ff1_d)
            ff2_d = tf.keras.layers.Dropout(0.2)(ff2)

            encoder_inputs = sublayer_connection()(encoder_inputs, ff2_d)

        return encoder_inputs

    def build_decoder_block(self, encoder_inputs, decoder_inputs, mask):
        for _ in range(self.hp.decoder_block):
            multihead_attention = self.attention(
                q = decoder_inputs,
                k = decoder_inputs,
                v = decoder_inputs,
                num_heads = self.hp.num_heads,
                head_dim = self.hp.head_dims,
                mask = mask,
            )

            decoder_inputs = sublayer_connection()(decoder_inputs, multihead_attention)

            multihead_attention = self.attention(
                q = encoder_inputs,
                k = encoder_inputs,
                v = decoder_inputs,
                num_heads = self.hp.num_heads,
                head_dim = self.hp.head_dims,
                mask = None,
            )

            decoder_inputs = sublayer_connection()(decoder_inputs, multihead_attention)

            ff1 = tf.keras.layers.Dense(self.hp.num_heads * self.hp.head_dims * 4)(decoder_inputs)
            ff1_d = tf.keras.layers.Dropout(0.2)(ff1)
            ff2 = tf.keras.layers.Dense(self.hp.num_heads * self.hp.head_dims)(ff1_d)
            ff2_d = tf.keras.layers.Dropout(0.2)(ff2)

            decoder_inputs = sublayer_connection()(decoder_inputs, ff2_d)

        return decoder_inputs







