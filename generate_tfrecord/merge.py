# python generate_tfrecord/merge.py --out_name ijoon_251015.tfrecord --tfrecord_list /notebooks/datasets/face/trillionpairs_vggface2_100979.tfrecord,/notebooks/datasets/face/aihub_korean_face/aihub_korean_face.tfrecord,/notebooks/datasets/face/AFD/AFDB_dataset_160_part1.tfrecord,/notebooks/datasets/face/hanbat_face/output_folder_clean_2.tfrecord --labels 100980,400,1662,259 --approx_totals 5382696,919896,310969,5784
import argparse, os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU (only read/write is allowed so CPU is safe)

import tensorflow as tf
from tqdm import tqdm

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _parse_tfrecord(serialized):
    desc = {
        'jpeg': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'x1': tf.io.FixedLenFeature((), tf.int64),
        'y1': tf.io.FixedLenFeature((), tf.int64),
        'x2': tf.io.FixedLenFeature((), tf.int64),
        'y2': tf.io.FixedLenFeature((), tf.int64),
    }
    return tf.io.parse_single_example(serialized, desc)

def _count_examples(tfrec_path, approx=None, probe_n=0):
    """accurate count (slow) or probe estimation (fast)."""
    if approx is not None:
        return approx
    if probe_n and probe_n > 0:
        # file size / average record size
        import os
        sizes = []
        try:
            it = tf.compat.v1.io.tf_record_iterator(path=tfrec_path)
            for i, rec in enumerate(it):
                sizes.append(len(rec))
                if i+1 >= probe_n:
                    break
        except Exception:
            sizes = []
        if sizes:
            avg = sum(sizes) / len(sizes)
            return max(int(os.path.getsize(tfrec_path) / avg), len(sizes))
    # accurate count
    c = 0
    for _ in tf.data.TFRecordDataset(tfrec_path):
        c += 1
    return c

def load_all_examples_to_memory(tfrecord_list, max_labels, approx_totals=None, probe_n=0):
    """
    load all examples from all TFRecord files into memory and fully shuffle
    """
    if len(tfrecord_list) != len(max_labels):
        raise ValueError("tfrecord_list and max_labels have different lengths.")
    if approx_totals and (len(approx_totals) != len(tfrecord_list)):
        raise ValueError("approx_totals has different length from tfrecord_list.")

    all_examples = []
    label_accum = 0

    print("🔄 Loading all TFRecord files into memory...")
    
    for idx, (path, max_label) in enumerate(zip(tfrecord_list, max_labels)):
        print(f"📁 Loading {path}... (label offset: {label_accum} .. {label_accum + max_label})")
        
        # count examples per file
        approx = None if not approx_totals else approx_totals[idx]
        file_count = _count_examples(path, approx=approx, probe_n=probe_n)
        
        # load all examples from the file into memory
        ds = tf.data.TFRecordDataset(path).map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        
        local_examples = []
        for ex in tqdm(ds, total=file_count, desc=f"Loading: {os.path.basename(path)}"):
            # apply offset
            glabel = ex['label'] + int(label_accum)
            
            # convert example to dictionary and store in memory
            example_dict = {
                'jpeg': ex['jpeg'].numpy(),
                'label': int(glabel.numpy()),
                'x1': int(ex['x1'].numpy()),
                'y1': int(ex['y1'].numpy()),
                'x2': int(ex['x2'].numpy()),
                'y2': int(ex['y2'].numpy()),
                'source_file': path,  # for debugging
                'original_label': int(ex['label'].numpy())  # for debugging
            }
            local_examples.append(example_dict)
        
        all_examples.extend(local_examples)
        label_accum += (int(max_label) + 1)
        
        print(f"  ✅ {len(local_examples):,} examples loaded")

    print(f"🎯 Total {len(all_examples):,} examples loaded")
    
    # fully shuffle
    print("🔀 Fully shuffling...")
    random.shuffle(all_examples)
    print("✅ Shuffling completed!")
    
    return all_examples

def merge_shuffled(out_name, tfrecord_list, max_labels, approx_totals=None, probe_n=0, seed=42):
    """
    create TFRecord from fully shuffled data
    """
    # set seed
    random.seed(seed)
    
    # load all examples into memory and shuffle
    all_examples = load_all_examples_to_memory(tfrecord_list, max_labels, approx_totals, probe_n)
    
    # save shuffled examples to TFRecord
    writer = tf.io.TFRecordWriter(out_name)
    
    print(f"💾 Saving {len(all_examples):,} shuffled examples to TFRecord...")
    
    for i, ex in enumerate(tqdm(all_examples, desc="📊 TFRecord 저장")):
        feat = {
            'jpeg': _bytes_feature(ex['jpeg']),
            'label': _int64_feature(ex['label']),
            'x1': _int64_feature(ex['x1']),
            'y1': _int64_feature(ex['y1']),
            'x2': _int64_feature(ex['x2']),
            'y2': _int64_feature(ex['y2']),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feat))
        writer.write(example.SerializeToString())
    writer.close()
    print(f"✅ Completed! Total {len(all_examples):,} examples shuffled and saved to {out_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_name', required=True)
    ap.add_argument('--tfrecord_list', required=True,
                    help='ex) file1.tfrecord,file2.tfrecord')
    ap.add_argument('--labels', required=True,
                    help='Maximum label value for each TFRecord. ex) 5748,100980 (0..max assumed)')
    ap.add_argument('--approx_totals', default=None,
                    help='Number of samples per file (if known). ex) 13150,5382696')
    ap.add_argument('--probe_n', type=int, default=0,
                    help='Number of samples to use for total count estimation by sampling (0 means not used)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random shuffle seed (default: 42)')
    args = ap.parse_args()

    files = args.tfrecord_list.split(',')
    maxlabs = [int(x) for x in args.labels.split(',')]
    approx = None
    if args.approx_totals:
        approx = [int(x) for x in args.approx_totals.split(',')]

    merge_shuffled(args.out_name, files, maxlabs, approx_totals=approx, probe_n=args.probe_n, seed=args.seed)
