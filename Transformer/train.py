from module.model import Transformer
from HyperParameter.transformer import TransformerHyperParameter
import tensorflow as tf




hp = TransformerHyperParameter()

transformer = Transformer(hp)

model = transformer.get_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()