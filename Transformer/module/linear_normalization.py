import tensorflow as tf

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        # γ (Gamma)와 β (Beta) 파라미터 생성
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],  # 마지막 차원 기준 (채널 수)
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)  # 마지막 차원 기준 평균
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)  # 분산 계산
        norm_x = (x - mean) / tf.sqrt(variance + self.epsilon)  # 정규화
        return self.gamma * norm_x + self.beta