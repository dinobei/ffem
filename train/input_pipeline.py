import tensorflow as tf


TF_AUTOTUNE = tf.data.AUTOTUNE


def random_flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)


def gray(x):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))


def random_color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.1)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x


def blur(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def gfilter(x):
        # Use native TensorFlow gaussian filter
        return tf.nn.depthwise_conv2d(
            tf.expand_dims(x, 0),
            tf.reshape(tf.constant([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]), [3, 3, 1, 1]),
            strides=[1, 1, 1, 1],
            padding='SAME'
        )[0] / 16.0

    def mfilter(x):
        # Simple median-like filter using average pooling as approximation
        return tf.nn.avg_pool2d(tf.expand_dims(x, 0), ksize=3, strides=1, padding='SAME')[0]

    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))


def cutout(x: tf.Tensor):
    def _cutout(x: tf.Tensor):
        h = tf.shape(x)[0]
        w = tf.shape(x)[1]
        size = tf.random.uniform([], 0, 20, dtype=tf.int32) * 2
        size = tf.minimum(size, tf.minimum(h, w))
        if size <= 0:
            return x
        y = tf.random.uniform([], 0, h - size + 1, dtype=tf.int32)
        x_pos = tf.random.uniform([], 0, w - size + 1, dtype=tf.int32)
        # 2D 마스크 생성
        mask2d = tf.ones([h, w], dtype=x.dtype)
        mask2d = tf.tensor_scatter_nd_update(
            mask2d,
            tf.reshape(tf.stack(tf.meshgrid(
                tf.range(y, y + size), tf.range(x_pos, x_pos + size), indexing='ij'), axis=-1), [-1, 2]),
            tf.zeros([size * size], dtype=x.dtype)
        )
        # 3D로 확장
        mask3d = tf.expand_dims(mask2d, axis=-1)
        mask3d = tf.tile(mask3d, [1, 1, tf.shape(x)[-1]])
        return x * mask3d
    choice = tf.random.uniform([], 0., 1., dtype=tf.float32)
    return tf.cond(choice > 0.5, lambda: _cutout(x), lambda: x)


def make_tfdataset(train_tfrecord, test_tfrecord, batch_size, img_shape):
    train_ds = tf.data.TFRecordDataset(train_tfrecord)

    def _read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.int64),
            'y1': tf.io.FixedLenFeature((), tf.int64),
            'x2': tf.io.FixedLenFeature((), tf.int64),
            'y2': tf.io.FixedLenFeature((), tf.int64)
        }
        example = tf.io.parse_single_example(serialized, description)
        image = tf.io.decode_jpeg(example['jpeg'], channels=3)
        label = example['label']
        box = [
            tf.cast(example['y1'], tf.float32),
            tf.cast(example['x1'], tf.float32),
            tf.cast(example['y2'], tf.float32),
            tf.cast(example['x2'], tf.float32)]
        return image, label, box

    def _load_and_preprocess_image(image, label, box):
        # shape = [Height, Width, Channel]
        shape = tf.shape(image)
        # shape = [Height, Height, Width, Width]
        shape = tf.repeat(shape, [2, 2, 0])
        # shape = [Height, Width, Height, Width]
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        # Normalize [y1, x1, y2, x2] box by width and height.
        box /= tf.cast(shape, tf.float32)
        image = tf.cast(image, tf.float32)
        return image, label, box


    def _random_crop(x: tf.Tensor, label, box):
        def crop_rnd_wrap(x, box):
            scale = tf.random.uniform([4], -0.1, 0.1)
            box += box * scale
            box = tf.clip_by_value(box, 0, 1)
            return tf.image.crop_and_resize([x], [box], [0], img_shape)[0]


        def crop_wrap(x, box):
            return tf.image.crop_and_resize([x], [box], [0], img_shape)[0]


        choice = tf.random.uniform(shape=[], minval=0., maxval=1.)
        # Only apply cropping 50% of the time
        cond = tf.cond(choice < 0.5,
            lambda: crop_wrap(x, box), lambda: crop_rnd_wrap(x, box))
        return cond, label


    def _normalize(x: tf.Tensor):
        # Normalize images to the range [0, 1].
        return x / 255.

    train_ds = train_ds.map(_read_tfrecord)
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.map(_load_and_preprocess_image, num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.map(_random_crop, num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda img, label : (_normalize(img), label), num_parallel_calls=TF_AUTOTUNE)
    augmentations = [random_flip, random_color]
    for f in augmentations:
        choice = tf.random.uniform([], 0.0, 1.0)
        train_ds = train_ds.map(lambda x, label: (tf.cond(choice > 0.5, lambda: f(x), lambda: x), label),
            num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.map(lambda x, label: (tf.clip_by_value(x, 0., 1.), label), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.map(lambda x, label: (cutout(x), label), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds_dict = {}
    for test_file in test_tfrecord:
        test_ds = tf.data.TFRecordDataset(test_file)
        test_ds = test_ds.map(_read_tfrecord)
        test_ds = test_ds.map(_load_and_preprocess_image, num_parallel_calls=TF_AUTOTUNE)
        test_ds = test_ds.map(
            lambda x, label, box: (tf.image.crop_and_resize([x], [box], [0], img_shape)[0], label),
            num_parallel_calls=TF_AUTOTUNE)
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.map(lambda img, label : (_normalize(img), label), num_parallel_calls=TF_AUTOTUNE)
        test_ds_dict[test_file] = test_ds

    return train_ds, test_ds_dict
