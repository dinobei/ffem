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
        # Gaussian filter using native TensorFlow
        # Create a 5x5 Gaussian kernel
        kernel = tf.constant([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=tf.float32) / 256.0
        
        # Reshape kernel for depthwise convolution
        kernel = tf.reshape(kernel, [5, 5, 1, 1])
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        
        # Apply convolution
        x_expanded = tf.expand_dims(x, 0)  # Add batch dimension
        x_filtered = tf.nn.depthwise_conv2d(x_expanded, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return x_filtered[0]  # Remove batch dimension

    def mfilter(x):
        # Median filter approximation using average pooling
        # This is a simplified approximation since true median filter is complex
        x_expanded = tf.expand_dims(x, 0)  # Add batch dimension
        x_pooled = tf.nn.avg_pool2d(x_expanded, ksize=3, strides=1, padding='SAME')
        return x_pooled[0]  # Remove batch dimension

    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))


def cutout(x: tf.Tensor):
    def _cutout_single_image(img: tf.Tensor):
        # Get image dimensions for single image
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        c = tf.shape(img)[2]
        
        # Generate random size
        size = tf.random.uniform([], 0, 20, dtype=tf.int32) * 2
        size = tf.minimum(size, tf.minimum(h, w))
        
        # Check if size is valid
        def apply_cutout():
            # Generate random position
            y = tf.random.uniform([], 0, h - size + 1, dtype=tf.int32)
            x_pos = tf.random.uniform([], 0, w - size + 1, dtype=tf.int32)
            
            # Create 2D mask for single image
            mask2d = tf.ones([h, w], dtype=img.dtype)
            
            # Create indices for the cutout region
            y_range = tf.range(y, y + size)
            x_range = tf.range(x_pos, x_pos + size)
            
            # Create meshgrid
            y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing='ij')
            
            # Flatten and stack indices
            y_flat = tf.reshape(y_grid, [-1])
            x_flat = tf.reshape(x_grid, [-1])
            
            # Stack indices
            indices = tf.stack([y_flat, x_flat], axis=1)
            
            # Create zeros for the cutout region
            zeros = tf.zeros([tf.shape(indices)[0]], dtype=img.dtype)
            
            # Apply cutout mask
            mask2d = tf.tensor_scatter_nd_update(mask2d, indices, zeros)
            
            # Expand to 3D and broadcast to match image shape
            mask3d = tf.expand_dims(mask2d, axis=-1)
            mask3d = tf.broadcast_to(mask3d, [h, w, c])
            
            return img * mask3d
        
        def return_original():
            return img
        
        # Apply cutout only if size > 0
        return tf.cond(size > 0, apply_cutout, return_original)
    
    def _cutout_batch(batch_img: tf.Tensor):
        # Apply cutout to each image in the batch
        return tf.map_fn(_cutout_single_image, batch_img)
    
    choice = tf.random.uniform([], 0., 1., dtype=tf.float32)
    return tf.cond(choice > 0.5, lambda: _cutout_batch(x), lambda: x)


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
    train_ds = train_ds.repeat()  # Add repeat back for infinite dataset, controlled by steps_per_epoch
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
