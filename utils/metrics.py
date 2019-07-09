import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)#########new version
    mask = tf.cast(mask, dtype=tf.float32)###转换数据类型
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def softmax_cross_entropy(preds, labels):
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)############# new version
    return tf.reduce_mean(loss)

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def micro_F1_score(preds, labels):
    '''
    考虑到类别存在不均衡，采用micro f1 而不是macro f1
    '''
    preds = tf.one_hot(tf.argmax(preds,1), depth = preds.shape[1])#, dtype = tf.int32
    number = tf.shape(preds)

    TP = tf.count_nonzero(    preds   *  labels,       axis = None)
    FP = tf.count_nonzero(    preds   *  (labels-1),   axis = None)
    FN = tf.count_nonzero(  (preds-1) *  labels,       axis = None)
    TN = tf.count_nonzero(  (preds-1) *  (labels-1),   axis = None)

    precision = TP/(TP+FP)
    recall  = TP/(TP+FN)
    f1 = 2*precision*recall / (precision+recall)
    #micro_f1 = tf.reduce_mean(f1)
    return f1, (TP,FP,FN,TN), number[0]
