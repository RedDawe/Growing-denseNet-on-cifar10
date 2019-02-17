import pickle
import numpy as np
import tensorflow as tf
from math import isnan
import sys
import time
tf.set_random_seed(0)
np.random.seed(0)

batch_size = 100
real_low = 0.00001
real_high = 0.0001
exp_lr = True
momentum=0.99
init_from_files = False
save_every_cycle = False

eval_only = False

def hyper_parametes(real_low, real_high, batch_size):
    stepsize = 2 * 40000 / batch_size
    if exp_lr:
        low = 0
        high = np.log10(real_high) - np.log10(real_low)
    else:
        low = real_low
        high = real_high

    return stepsize, low, high
stepsize, low, high = hyper_parametes(real_low, real_high, batch_size)
def load_cfar10_batch(batch_id):
    if batch_id == 0:
        cifar10_dataset_folder_path = 'D:/cifar-10-batches-py/test_batch'
    else:
        cifar10_dataset_folder_path = 'D:/cifar-10-batches-py/data_batch_' + str(batch_id)
    with open(cifar10_dataset_folder_path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return np.array(features), np.array(labels)
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def get_data():
    X = []
    Y = []
    for batch_id in range(6):
        features, labels = load_cfar10_batch(batch_id)
        X.append(features)
        Y.append(labels)
    X = np.stack(X)
    Y = np.stack(Y)

    X_test = X[0, :, :,  :, :].reshape(-1, batch_size, 32, 32, 3)
    Y_test = Y[0, :].reshape(-1, batch_size)
    X_cv = X[-1, :, :, :, :].reshape(-1, batch_size, 32, 32, 3)
    Y_cv = Y[-1, :].reshape(-1, batch_size)
    X_train = X[1:-1, :, :, :, :].reshape(-1, batch_size, 32, 32, 3)
    Y_train = Y[1:-1, :].reshape(-1, batch_size)

    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
path = '/tmp/model6.ckpt'
tmp_path = '/tmp/model3.ckpt'


last_model = 0
growing = not eval_only
max_epochs = 100 * stepsize
n_l = 30
n_f = 12
rate = 0
best_n_l = n_l
best_n_f = n_f
n_l_step = 10
best_rate = rate
era = 0
max_eras = 200
while growing:
    print('Running n_l =', n_l, 'n_f =', n_f, 'rate = ', rate)

    tf.reset_default_graph()
    with tf.name_scope('network'):
        with tf.name_scope('data'):
            x_train = tf.data.Dataset.from_tensor_slices(tf.constant(X_train, dtype=tf.float32))
            y_train = tf.data.Dataset.from_tensor_slices(tf.constant(Y_train, dtype=tf.float32))
            x_cv = tf.data.Dataset.from_tensor_slices(tf.constant(X_cv, dtype=tf.float32))
            y_cv = tf.data.Dataset.from_tensor_slices(tf.constant(Y_cv, dtype=tf.float32))
            x_test = tf.data.Dataset.from_tensor_slices(tf.constant(X_test, dtype=tf.float32))
            y_test = tf.data.Dataset.from_tensor_slices(tf.constant(Y_test, dtype=tf.float32))

            x_iterator = tf.data.Iterator.from_structure(x_train.output_types, x_train.output_shapes)
            y_iterator = tf.data.Iterator.from_structure(y_train.output_types, y_train.output_shapes)

            next_x_element = x_iterator.get_next()
            next_y_element = y_iterator.get_next()

            x_train_init_op = x_iterator.make_initializer(x_train)
            x_cv_init_op = x_iterator.make_initializer(x_cv)
            x_test_init_op = x_iterator.make_initializer(x_test)
            y_train_init_op = y_iterator.make_initializer(y_train)
            y_cv_init_op = y_iterator.make_initializer(y_cv)
            y_test_init_op = y_iterator.make_initializer(y_test)

            epochCounter = tf.placeholder(tf.float32)
            training = tf.placeholder(tf.bool)

        a = next_x_element
        a_hat = next_y_element

        for dense_block in range(n_l):
            a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
            a_c = tf.layers.dropout(a_c, rate=rate, training=training)
            a_c = tf.layers.conv2d(a_c, filters=n_f, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1), name='to_be_restored/' + str(dense_block))
            a = tf.concat([a_c, a], axis=-1)

        a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a_c = tf.layers.dropout(a_c, rate=rate, training=training)
        a_c = tf.layers.conv2d(tf.pad(a_c, [[0, 0], [0, 1], [0, 1], [0, 0]]), filters=n_f, kernel_size=[3, 3], strides=[3, 3], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))
        a_p = tf.layers.max_pooling2d(tf.pad(a, [[0, 0], [0, 1], [0, 1], [0, 0]]), pool_size=[3, 3], strides=[3, 3], padding='valid')
        a = tf.concat([a_c, a_p], axis=-1)
        a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a_c = tf.layers.dropout(a_c, rate=rate, training=training)
        a_c = tf.layers.conv2d(tf.pad(a_c, [[0, 0], [0, 1], [0, 1], [0, 0]]), filters=n_f, kernel_size=[3, 3], strides=[3, 3], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))
        a_p = tf.layers.max_pooling2d(tf.pad(a, [[0, 0], [0, 1], [0, 1], [0, 0]]), pool_size=[3, 3], strides=[3, 3], padding='valid')
        a = tf.concat([a_c, a_p], axis=-1)
        a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a_c = tf.layers.dropout(a_c, rate=rate, training=training)
        a_c = tf.layers.conv2d(tf.pad(a_c, [[0, 0], [0, 0], [0, 0], [0, 0]]), filters=n_f, kernel_size=[2, 2], strides=[2, 2], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))
        a_p = tf.layers.max_pooling2d(a, pool_size=[2, 2], strides=[2, 2], padding='valid')
        a = tf.concat([a_c, a_p], axis=-1)
        a = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a = tf.layers.dropout(a, rate=rate, training=training)
        a = tf.layers.conv2d(tf.pad(a, [[0, 0], [0, 0], [0, 0], [0, 0]]), filters=10, kernel_size=[2, 2], strides=[2, 2], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))

        logits = tf.reshape(a, [batch_size, 10])
        labels = tf.reshape(tf.one_hot(tf.cast(a_hat, tf.int32), depth=10), [batch_size, 10])

        J = tf.losses.softmax_cross_entropy(labels, logits)
        accuracy = tf.reduce_mean(tf.where(tf.equal(logits, labels), tf.ones(tf.shape(logits)), tf.zeros(tf.shape(logits))))

        with tf.name_scope('learning_rate'):
            cycle = tf.floor(1 + epochCounter / 2 / stepsize)
            scale = tf.abs(epochCounter / tf.constant(stepsize, dtype=tf.float32) - 2 * cycle + 1)
            alpha = low + (high - low) * tf.maximum(tf.constant(0.0, dtype=tf.float32), (1 - scale))
            if exp_lr:
                alpha = real_low * 10 ** alpha

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(J)

        reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars if (var.op.name.split('/')[0] == 'to_be_restored' and int(var.op.name.split('/')[1]) < n_l - n_l_step) or var.op.name.split('/')[0] == 'Network'])
        if reuse_vars_dict:
            load_saver = tf.train.Saver(reuse_vars_dict)
        save_saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if init_from_files == True or era != 0:
            if reuse_vars_dict:
                load_saver.restore(sess, tmp_path)

        epoch = 0
        while epoch < (max_epochs if init_from_files == False and era == 0 else max_epochs / 4):
            costs = []
            naturally = False
            sess.run([x_train_init_op, y_train_init_op])
            start = time.time()
            for batch in range(X_train.shape[0]):
                _, cost = sess.run([optimizer, J], feed_dict={
                    epochCounter: epoch,
                    training: True
                })
                costs.append(cost)

                if isnan(cost):
                    print(epoch)
                    print('Got NaN as a cost. Ending the growing process.')
                    growing = False
                    break

                costs.append(cost)
            else:
                naturally = True
            if not naturally:
                break
            print(time.time()-start)

            if epoch % (2 * stepsize) == 0:
                performance = np.mean(costs)
                print()
                print(epoch, '-', performance)
                if save_every_cycle:
                    save_saver.save(sess, tmp_path)
            print(epoch, end=' ', flush=True)
            epoch += 1

        print()
        costs = []
        accs = []
        sess.run([x_cv_init_op, y_cv_init_op])
        for batch in range(X_cv.shape[0]):
            acc, cost = sess.run([accuracy, J], feed_dict={
                epochCounter: epoch,
                training: False
            })
            costs.append(cost)
            accs.append(acc)

        model = np.mean(accs)

        print(n_l, n_f, rate, 'Cv accuracy =', model)

        if model >= last_model:
            save_saver.save(sess, tmp_path)
            save_saver.save(sess, path)

            best_n_l = n_l
            best_rate = rate

            n_l += n_l_step

            last_model = model
        else:
            save_saver.save(sess, tmp_path)

            rate += 0.1


    era += 1

    if era == max_eras:
        growing = False


tf.reset_default_graph()
print('Best n_l =', best_n_l, 'and best n_f =', best_n_f)
n_f = best_n_f
n_l = best_n_l
with tf.Session(config=config) as sess:
    with tf.name_scope('network'):
        with tf.name_scope('data'):
            x_train = tf.data.Dataset.from_tensor_slices(tf.constant(X_train, dtype=tf.float32))
            y_train = tf.data.Dataset.from_tensor_slices(tf.constant(Y_train, dtype=tf.float32))
            x_cv = tf.data.Dataset.from_tensor_slices(tf.constant(X_cv, dtype=tf.float32))
            y_cv = tf.data.Dataset.from_tensor_slices(tf.constant(Y_cv, dtype=tf.float32))
            x_test = tf.data.Dataset.from_tensor_slices(tf.constant(X_test, dtype=tf.float32))
            y_test = tf.data.Dataset.from_tensor_slices(tf.constant(Y_test, dtype=tf.float32))

            x_iterator = tf.data.Iterator.from_structure(x_train.output_types, x_train.output_shapes)
            y_iterator = tf.data.Iterator.from_structure(y_train.output_types, y_train.output_shapes)

            next_x_element = x_iterator.get_next()
            next_y_element = y_iterator.get_next()

            x_train_init_op = x_iterator.make_initializer(x_train)
            x_cv_init_op = x_iterator.make_initializer(x_cv)
            x_test_init_op = x_iterator.make_initializer(x_test)
            y_train_init_op = y_iterator.make_initializer(y_train)
            y_cv_init_op = y_iterator.make_initializer(y_cv)
            y_test_init_op = y_iterator.make_initializer(y_test)

            epochCounter = tf.placeholder(tf.float32)
            training = tf.placeholder(tf.bool)

        a = next_x_element
        a_hat = next_y_element

        for dense_block in range(n_l):
            a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
            a_c = tf.layers.dropout(a_c, rate=rate, training=training)
            a_c = tf.layers.conv2d(a_c, filters=n_f, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1), name='to_be_restored/' + str(dense_block))
            a = tf.concat([a_c, a], axis=-1)

        a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a_c = tf.layers.dropout(a_c, rate=rate, training=training)
        a_c = tf.layers.conv2d(tf.pad(a_c, [[0, 0], [0, 1], [0, 1], [0, 0]]), filters=n_f, kernel_size=[3, 3], strides=[3, 3], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))
        a_p = tf.layers.max_pooling2d(tf.pad(a, [[0, 0], [0, 1], [0, 1], [0, 0]]), pool_size=[3, 3], strides=[3, 3], padding='valid')
        a = tf.concat([a_c, a_p], axis=-1)
        a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a_c = tf.layers.dropout(a_c, rate=rate, training=training)
        a_c = tf.layers.conv2d(tf.pad(a_c, [[0, 0], [0, 1], [0, 1], [0, 0]]), filters=n_f, kernel_size=[3, 3], strides=[3, 3], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))
        a_p = tf.layers.max_pooling2d(tf.pad(a, [[0, 0], [0, 1], [0, 1], [0, 0]]), pool_size=[3, 3], strides=[3, 3], padding='valid')
        a = tf.concat([a_c, a_p], axis=-1)
        a_c = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a_c = tf.layers.dropout(a_c, rate=rate, training=training)
        a_c = tf.layers.conv2d(tf.pad(a_c, [[0, 0], [0, 0], [0, 0], [0, 0]]), filters=n_f, kernel_size=[2, 2], strides=[2, 2], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))
        a_p = tf.layers.max_pooling2d(a, pool_size=[2, 2], strides=[2, 2], padding='valid')
        a = tf.concat([a_c, a_p], axis=-1)
        a = tf.layers.batch_normalization(a, training=training, momentum=momentum)
        a = tf.layers.dropout(a, rate=rate, training=training)
        a = tf.layers.conv2d(tf.pad(a, [[0, 0], [0, 0], [0, 0], [0, 0]]), filters=10, kernel_size=[2, 2], strides=[2, 2], padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.1))

        logits = tf.reshape(a, [batch_size, 10])
        labels = tf.reshape(tf.one_hot(tf.cast(a_hat, tf.int32), depth=10), [batch_size, 10])

        J = tf.losses.softmax_cross_entropy(labels, logits)
        accuracy = tf.reduce_mean(tf.where(tf.equal(logits, labels), tf.ones(tf.shape(logits)), tf.zeros(tf.shape(logits))))

        with tf.name_scope('learning_rate'):
            cycle = tf.floor(1 + epochCounter / 2 / stepsize)
            scale = tf.abs(epochCounter / tf.constant(stepsize, dtype=tf.float32) - 2 * cycle + 1)
            alpha = low + (high - low) * tf.maximum(tf.constant(0.0, dtype=tf.float32), (1 - scale))
            if exp_lr:
                alpha = real_low * 10 ** alpha

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(J)

        reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
        saver = tf.train.Saver(reuse_vars_dict)

    saver.restore(sess, path)

    costs = []
    accs = []
    sess.run([x_train_init_op, y_train_init_op])
    for batch in range(X_train.shape[0]):
        acc, cost = sess.run([accuracy, J], feed_dict={
            epochCounter: 0,
            training: False
        })
        costs.append(cost)
        accs.append(acc)
    print('Validation cost =', np.mean(costs), 'accuracy =', np.mean(accs))

    costs = []
    accs = []
    sess.run([x_cv_init_op, y_cv_init_op])
    for batch in range(X_cv.shape[0]):
        acc, cost = sess.run([accuracy, J], feed_dict={
            epochCounter: 0,
            training: False
        })
        costs.append(cost)
        accs.append(acc)
    print('Cross validation cost =', np.mean(costs), 'accuracy =', np.mean(accs))