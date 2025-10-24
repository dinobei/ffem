# python generate_tfrecord/main.py --root_path /notebooks/datasets/face/custom_250826 --json_file /notebooks/datasets/face/custom_250826.json --output generate_tfrecord/custom_250826.tfrecord

import argparse
import os
import json
import random

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecord(root_path, out_file, example_json, max_label=None, max_images_per_label=None):
    tf_file = tf.io.TFRecordWriter(out_file)
    class_max = 0
    count = 0
    label_counts = {}  # count the number of images per label
    
    for n, img_name in enumerate(example_json):
        data = example_json[img_name]
        if max_label is not None and data['label'] > max_label:
            continue
        
        # check the maximum number of images per label
        if max_images_per_label is not None:
            current_label = data['label']
            if current_label not in label_counts:
                label_counts[current_label] = 0
            
            if label_counts[current_label] >= max_images_per_label:
                continue  # skip if the number of images per label reaches the limit
        
        with open(os.path.join(root_path, img_name), 'rb') as jpeg_file:
            jpeg_bytes = jpeg_file.read()
        if jpeg_bytes is None:
            print('{} is skipped because it cannot read the file.'.format(img_name))
            continue
        feature = {
            'jpeg': _bytes_feature(jpeg_bytes),
            'label': _int64_feature(data['label']),
            'x1': _int64_feature(data['x1']),
            'y1': _int64_feature(data['y1']),
            'x2': _int64_feature(data['x2']),
            'y2': _int64_feature(data['y2'])
        }
        exam = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_file.write(exam.SerializeToString())
        count = count + 1
        
        # count the number of images after successfully saving the image
        if max_images_per_label is not None:
            current_label = data['label']
            label_counts[current_label] += 1
        if data['label'] > class_max:
            class_max = data['label']
        if n % 1000 == 0:
            print('{} images saved.'.format(n))
    tf_file.close()
    print('generating tfrecord is finished.')
    print('total number of images: {}'.format(count))
    print('maximum class label : {}'.format(class_max))
    print(' If the label starts from 0, the total classes is ', class_max + 1)
    
    # print the number of images per label
    if max_images_per_label is not None:
        print('\nLabel distribution:')
        for label in sorted(label_counts.keys()):
            print('Label {}: {} images'.format(label, label_counts[label]))


def _derive_output_path(base_output_path, split_name):
    if base_output_path.endswith('.tfrecord'):
        return base_output_path[:-9] + '_' + split_name + '.tfrecord'
    return base_output_path + '_' + split_name


def _write_tfrecord_from_list(root_path, out_file, example_json, img_names):
    tf_file = tf.io.TFRecordWriter(out_file)
    count = 0
    class_max = 0
    unique_labels = set()
    for n, img_name in enumerate(img_names):
        data = example_json[img_name]
        with open(os.path.join(root_path, img_name), 'rb') as jpeg_file:
            jpeg_bytes = jpeg_file.read()
        if jpeg_bytes is None:
            print('{} is skipped because it cannot read the file.'.format(img_name))
            continue
        feature = {
            'jpeg': _bytes_feature(jpeg_bytes),
            'label': _int64_feature(data['label']),
            'x1': _int64_feature(data['x1']),
            'y1': _int64_feature(data['y1']),
            'x2': _int64_feature(data['x2']),
            'y2': _int64_feature(data['y2'])
        }
        exam = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_file.write(exam.SerializeToString())
        count += 1
        unique_labels.add(int(data['label']))
        if data['label'] > class_max:
            class_max = data['label']
        if n % 1000 == 0:
            print('{} images saved.'.format(n))
    tf_file.close()
    print('generating tfrecord is finished for {}.'.format(out_file))
    print('total number of images: {}'.format(count))
    print('maximum class label : {}'.format(class_max))
    print(' If the label starts from 0, the total classes is ', class_max + 1)
    print(' unique labels in this split: {}'.format(len(unique_labels)))


def _split_counts(total_items, ratios):
    exact = [ratios[0] * total_items, ratios[1] * total_items, ratios[2] * total_items]
    floors = [int(x) for x in [int(exact[0]), int(exact[1]), int(exact[2])]]
    # ensure floors are indeed floor
    floors = [int(x) for x in [float(exact[0]) // 1, float(exact[1]) // 1, float(exact[2]) // 1]]
    floors = [int(x) for x in floors]
    remainder = total_items - sum(floors)
    fracs = [exact[i] - floors[i] for i in range(3)]
    order = sorted(range(3), key=lambda i: fracs[i], reverse=True)
    for i in range(remainder):
        floors[order[i % 3]] += 1
    return floors


def _parse_split_arg(split_str):
    parts = [p.strip() for p in split_str.split(',')]
    if len(parts) != 3:
        return None
    try:
        ratios = [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        return None
    return ratios


def _validate_ratios(ratios):
    if any(r < 0.0 for r in ratios):
        return False
    total = sum(ratios)
    return abs(total - 1.0) <= 1e-6


def _parse_max_sample_arg(max_sample_str, is_split_mode):
    if max_sample_str is None:
        return None
    parts = [p.strip() for p in max_sample_str.split(',')]
    # non-split mode must be a single integer
    if not is_split_mode:
        if len(parts) != 1:
            print('Warning: --max_sample must be a single non-negative integer when not using --split. Exiting.')
            raise SystemExit(1)
        try:
            val = int(parts[0])
        except Exception:
            print('Warning: --max_sample must be an integer. Exiting.')
            raise SystemExit(1)
        if val < 0:
            print('Warning: --max_sample must be >= 0. Exiting.')
            raise SystemExit(1)
        return val
    # split mode: allow single int (apply to all) or three ints (train,val,test)
    if len(parts) == 1:
        try:
            val = int(parts[0])
        except Exception:
            print('Warning: --max_sample must be an integer or three integers separated by commas. Exiting.')
            raise SystemExit(1)
        if val < 0:
            print('Warning: --max_sample must be >= 0. Exiting.')
            raise SystemExit(1)
        return [val, val, val]
    if len(parts) == 3:
        try:
            vals = [int(parts[0]), int(parts[1]), int(parts[2])]
        except Exception:
            print('Warning: --max_sample must be an integer or three integers separated by commas. Exiting.')
            raise SystemExit(1)
        if any(v < 0 for v in vals):
            print('Warning: --max_sample values must be >= 0. Exiting.')
            raise SystemExit(1)
        return vals
    print('Warning: --max_sample must be an integer or three integers separated by commas. Exiting.')
    raise SystemExit(1)


def make_tfrecord_splits(root_path, base_output, example_json, ratios, seed=42, max_label=None, max_images_per_label=None, per_split_max_per_label=None):
    # group images by label with optional max_label filter
    label_to_images = {}
    for img_name, data in example_json.items():
        if max_label is not None and data['label'] > max_label:
            continue
        label = data['label']
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(img_name)

    rng = random.Random(seed)
    split_names = ['train', 'val', 'test']
    split_lists = {name: [] for name in split_names}

    # assign whole labels to splits (no overlap of the same label across splits)
    labels = list(label_to_images.keys())
    rng.shuffle(labels)
    num_labels = len(labels)
    c_train, c_val, c_test = _split_counts(num_labels, ratios)
    # Build label slices per split
    label_slices = {
        'train': labels[0:c_train],
        'val': labels[c_train:c_train + c_val],
        'test': labels[c_train + c_val:c_train + c_val + c_test]
    }

    print('Split label counts (by assignment): train={}, val={}, test={}'.format(len(label_slices['train']), len(label_slices['val']), len(label_slices['test'])))

    # Prepare caps per split (0 means unlimited)
    caps = per_split_max_per_label if per_split_max_per_label is not None else [None, None, None]
    split_to_cap = {
        'train': (None if caps[0] in (None, 0) else caps[0]),
        'val': (None if caps[1] in (None, 0) else caps[1]),
        'test': (None if caps[2] in (None, 0) else caps[2])
    }

    for split_name in split_names:
        if ratios[split_names.index(split_name)] <= 0.0:
            print('Skipping {} split due to zero ratio.'.format(split_name))
            continue
        selected_labels = label_slices[split_name]
        if len(selected_labels) == 0:
            print('Skipping {} split because it has no labels.'.format(split_name))
            continue
        per_label_cap = split_to_cap[split_name]
        for label in selected_labels:
            img_list = list(label_to_images[label])
            rng.shuffle(img_list)
            # apply cap: prefer per-split cap; fall back to global max_images_per_label
            if per_label_cap is not None:
                img_list = img_list[:per_label_cap]
            elif max_images_per_label is not None:
                img_list = img_list[:max_images_per_label]
            if len(img_list) == 0:
                continue
            split_lists[split_name].extend(img_list)

    for name in split_names:
        img_names = split_lists[name]
        if len(img_names) == 0:
            print('Skipping {} split because it has no samples.'.format(name))
            continue
        out_path = _derive_output_path(base_output, name)
        print('Writing {} samples to {} ...'.format(len(img_names), out_path))
        _write_tfrecord_from_list(root_path, out_path, example_json, img_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
        help='absolute path of images in json_file')
    parser.add_argument('--json_file', type=str, required=True,
        help='examples including image relative path, label and bounding box')
    parser.add_argument('--output', type=str, required=True,
        help='tfrecord file name including extension')
    parser.add_argument('--max_label', type=int, required=False,
        default=None, help='maximum label')
    parser.add_argument('--max_images_per_label', type=int, required=False,
        default=None, help='maximum number of images per label')
    parser.add_argument('--split', type=str, required=False, default=None,
        help='comma-separated ratios for train,val,test (e.g., 0.8,0.1,0.1). Use 0 to skip a split.')
    parser.add_argument('--seed', type=int, required=False, default=42,
        help='random seed used for shuffling before splitting')
    parser.add_argument('--max_sample', type=str, required=False, default=None,
        help='per-ID max samples. Non-split: single int (e.g., 15). Split: either single int (applies to train/val/test) or three ints like 0,10,10 (0 disables limit for that split).')
    args = parser.parse_args()


    with open(args.json_file, 'r') as f:
        data = json.loads(f.read())

    if args.split is None:
        # Non-split mode: --max_sample must be a single int if provided
        effective_cap = None
        if args.max_sample is not None:
            cap = _parse_max_sample_arg(args.max_sample, is_split_mode=False)
            effective_cap = None if cap == 0 else cap
        else:
            effective_cap = args.max_images_per_label
        make_tfrecord(args.root_path, args.output, data, args.max_label, effective_cap)
    else:
        ratios = _parse_split_arg(args.split)
        if ratios is None or not _validate_ratios(ratios):
            print('Warning: split ratios must be three non-negative numbers that sum to 1. Exiting.')
            raise SystemExit(1)
        # Split mode: parse per-split max_sample if provided
        per_split_caps = None
        if args.max_sample is not None:
            caps = _parse_max_sample_arg(args.max_sample, is_split_mode=True)
            per_split_caps = caps  # list of three ints
        make_tfrecord_splits(
            args.root_path,
            args.output,
            data,
            ratios,
            seed=args.seed,
            max_label=args.max_label,
            max_images_per_label=(None if per_split_caps is not None else args.max_images_per_label),
            per_split_max_per_label=per_split_caps
        )
