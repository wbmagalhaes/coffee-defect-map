import tensorflow as tf

K = tf.keras.backend


class IoU(tf.keras.metrics.Mean):
    def __init__(
            self,
            smooth=1.,
            name='iou',
            dtype=None):
        super(IoU, self).__init__(
            name=name,
            dtype=dtype)
        self.smooth = smooth

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        values = (intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + self.smooth)

        return super(IoU, self).update_state(values, sample_weight=sample_weight)

    def get_config(self):
        config = super(IoU, self).get_config()
        config.update({'smooth': self.smooth})
        return config
