def batch_encode(tokenizer, texts):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.


    Input:
        - tokenizer:  tokenizer object from the PreTrainedTokenizer Class
        - texts:  a list of strings where each string represents a piece of text
    Output:
        - input_ids:       a text encoded as a tf.Tensor object
        - attention_mask:  the text's attention mask encoded as a tf.Tensor object
    """""""""
    inputs = tokenizer.batch_encode_plus(texts,
                                         max_length=params['MAX_LENGTH'],
                                         padding='longest',  # implements dynamic padding
                                         truncation=True,
                                         return_tensors='tf',
                                         return_attention_mask=True,
                                         return_token_type_ids=False)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    return input_ids, attention_mask


def focal_loss(gamma=params['FL_GAMMA'], alpha=params['FL_ALPHA']):
    """""""""
    Function that computes the focal loss.
    
    Code adapted from https://gist.github.com/mkocabas/62dcd2f14ad21f3b25eac2d39ec2cc95
    """""""""
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
