import tensorflow as tf


def get_style_loss(feature_im,
                   target_im):
    """
    Expects two images shape (h,w,c)
    Args:
        feature_im: shape h, w, c
        target_im: shpae h, w, c
    Return:
        style loss (scaler)
    """
    loss = tf.reduce_sum(tf.square(feature_im - \
                  target_im))
    return loss



def get_content_loss(feature_im,
                   target_im):
    """
    Expects two images shape (h,w,c)
    Args:
        feature_im: shape h, w, c
        target_im: shpae h, w, c
    Return:
        content loss (scaler)
    """
    loss = 0.5 * (tf.reduce_sum(tf.square(feature_im - \
                  target_im)))
    return loss
