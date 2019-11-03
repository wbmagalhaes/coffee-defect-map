import tensorflow as tf

K = tf.keras.backend


class JaccardDistance(tf.keras.losses.Loss):
    def __init__(
            self,
            smooth=100,
            name='jaccard_distance'):
        super(JaccardDistance, self).__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        return (1 - jac) * self.smooth
