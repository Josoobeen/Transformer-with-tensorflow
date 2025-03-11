import tensorflow as tf

def create_padding_mask(seq):
    """
    padding이 되어 있는 부분을 찾아서 mask를 만드는 함수
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  # 패딩 부분을 0으로 바꿔준 뒤, 타입을 float32로 변환합니다.
    return seq[:, tf.newaxis, tf.newaxis, :]  # 해당 부분을 4차원 텐서로 변환하여 반환합니다.

def create_look_ahead_mask(size):
    """
    모델이 각 단어를 예측할 때, 이후 단어에 대한 접근을 막는 mask를 만드는 함수
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # tf.linalg.band_part 함수를 사용하여 아래쪽 삼각형 부분을 0으로 만듭니다.
    return mask[tf.newaxis, tf.newaxis, :, :]  # 2차원의 마스크를 반환합니다.
