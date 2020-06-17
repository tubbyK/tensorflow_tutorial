import tensorflow as tf

class tutorial_1():
    def __init__(self):
        pass

    def basic_tensors(self):
        string = tf.Variable('this is a string', tf.string)
        number = tf.Variable(324, tf.int64)
        floating = tf.Variable(3.567, tf.float64)
        return [string, number, floating]

    def rank_tensors(self):
        rank1_tensor = tf.Variable(['Test'], tf.string)
        rank2_tensor = tf.Variable([['hello', 'world'], ['bye', 'bye']], tf.string)
        return [rank1_tensor, rank2_tensor]

    def determine_rank(self, tensor):
        return tf.rank(tensor)

    def determine_shape(selfself, tensor):
        return tensor.shape

    def ones(self):
        tensor1 = tf.ones([1, 2, 3])
        tensor2 = tf.reshape(tensor1, [2, 3, 1])
        tensor3 = tf.reshape(tensor2, [3, -1])
        return tensor1, tensor2, tensor3

if __name__ == '__main__':
    print(tf.version)
    t = tutorial_1()

    # initialize tensors
    basic = t.basic_tensors()
    rank = t.rank_tensors()

    # print tensors
    print('tensors')
    [print(_) for _ in basic]
    [print(_) for _ in rank]

    # print rank of tensors
    print('\nrank of tensors')
    [print(t.determine_rank(_)) for _ in basic]
    [print(t.determine_rank(_)) for _ in rank]

    # print shape of tensors
    print('\nshape of tensors')
    [print(t.determine_shape(_)) for _ in basic]
    [print(t.determine_shape(_)) for _ in rank]

    # ones
    print('\nones')
    tensor1, tensor2, tensor3 = t.ones()
    [print(_) for _ in [tensor1, tensor2, tensor3]]