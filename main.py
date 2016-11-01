import tensorflow as tf


P_SIZE = 16
KEY_SIZE = 16
BATCH_SIZE = 4096
N = P_SIZE + KEY_SIZE


def l1_distance(p, d):
    return tf.reduce_sum(
        tf.abs(tf.sub(p, d)),
        reduction_indices=1
    )

def eve_penality(p, de):
    return tf.square(P_SIZE/2 - l1_distance(p, de)) / (P_SIZE/2) ** 2


def cypher_loss_function(p, db, de):
    """The loss function of Alice and Bob. Compute from the original plain text
    p, the decyphered db of Bob and the decyphered de of Eve."""
    return l1_distance(p, db)/P_SIZE + eve_penality(p, de)

def attacker_loss_function(p, de):
    return l1_distance(p, de)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def build_cypher_network(x, name, initial_size):
    net = tf.matmul(x, weight_variable([initial_size, N], name="{}/fc_layer".format(name)))
    net = tf.reshape(net, [BATCH_SIZE, N, 1])
    net = tf.sigmoid(tf.nn.conv1d(net, weight_variable([4, 1, 2], name="{}/conv1d-1".format(name)), stride=1, padding="SAME"))
    net = tf.sigmoid(tf.nn.conv1d(net, weight_variable([2, 2, 4], name="{}/conv1d-2".format(name)), stride=2, padding="SAME"))
    net = tf.sigmoid(tf.nn.conv1d(net, weight_variable([1, 4, 4], name="{}/conv1d-3".format(name)), stride=1, padding="SAME"))
    net = tf.tanh(tf.nn.conv1d(net, weight_variable([1, 4, 1], name="{}/conv1d-4".format(name)), stride=1, padding="SAME"))
    net = tf.reshape(net, [BATCH_SIZE, P_SIZE])
    return net

if __name__ == '__main__':
    sess = tf.InteractiveSession()

    # Generate a shared key between Alice and Bob
    k = 2 * tf.random_uniform([BATCH_SIZE, KEY_SIZE], minval=0, maxval=2, dtype=tf.int32) - 1
    k = tf.to_float(k)

    # Alice input: plaintext
    p = 2 * tf.random_uniform([BATCH_SIZE, P_SIZE], minval=0, maxval=2, dtype=tf.int32) - 1
    p = tf.to_float(p)
    x = tf.concat(1, [p, k])

    # Alice output and Bob and Eve input: cypher text
    c = tf.placeholder(tf.float32, shape=[BATCH_SIZE, P_SIZE])

    # Bob output: decyphered text, should be equal to p
    db = tf.placeholder(tf.float32, shape=[BATCH_SIZE, P_SIZE])

    # Eve output: decyphered text, should be equal to p
    de = tf.placeholder(tf.float32, shape=[BATCH_SIZE, P_SIZE])

    c = build_cypher_network(x, name="cypher/a", initial_size=N)
    db = build_cypher_network(tf.concat(1, [c, k]), name="cypher/b", initial_size=N)
    de = build_cypher_network(c, name="attacker", initial_size=P_SIZE)


    optimizer = tf.train.AdamOptimizer(0.0008)

    cypher_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cypher/")
    attacker_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "attacker/")

    cypher_train_step = optimizer.minimize(cypher_loss_function(p, db, de), var_list=cypher_vars)
    attacker_train_step = optimizer.minimize(attacker_loss_function(p, de), var_list=attacker_vars)

    cypher_accuracy = tf.reduce_mean(l1_distance(p, db))
    attacker_accuracy = tf.reduce_mean(attacker_loss_function(p, de))

    sess.run(tf.initialize_all_variables())
    for i in range(20000):
      if i % 100 == 0:
        train_accuracy = cypher_accuracy.eval(), attacker_accuracy.eval()
        print("step {}, training accuracy {}".format(i, train_accuracy))
      cypher_train_step.run()
      attacker_train_step.run()
      attacker_train_step.run()
