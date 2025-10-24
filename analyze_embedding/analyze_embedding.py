# echo "epoch77-vggface";python analyze_embedding/analyze_embedding.py --root /notebooks/datasets/face/vggface2/test --json /notebooks/datasets/face/vggface2/vggface2_test.json --ffem_model checkpoints/ResNet50_adaface_251001_2/best_inference_77.tflite --attr_model attr_ep150_wrapped_250626_128.tflite --yolo_face yolov8n-face.pt --use_sampling
# echo "epoch77-lfw";python analyze_embedding/analyze_embedding.py --root /notebooks/datasets/face/lfw/lfw_home/lfw_funneled/ --json generate_bbox/lfw_verified_bbox_250911.json --ffem_model checkpoints/ResNet50_adaface_251001_2/best_inference_77.tflite --attr_model attr_ep150_wrapped_250626_128.tflite --yolo_face yolov8n-face.pt
import os
import json
import argparse
import hashlib
from itertools import combinations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from ultralytics import YOLO
from ffem_embedder import FFEMTfliteEmbedder, FFEMKerasEmbedder

import face_identification as fid

# ─────────── utility functions ────────────

def hash_file(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# ─────────── JSON-based embedding ────────────

def load_json_entries(json_path, root):
    with open(json_path, 'r') as f:
        data = json.load(f)
    entries = []
    for rel, info in data.items():
        abs_path = os.path.join(root, rel)
        if os.path.isfile(abs_path):
            entries.append((abs_path, info))
    return entries

def _select_embedder(ffem_model, input_shape=(112,112)):
    lower = ffem_model.lower()
    if lower.endswith('.tflite'):
        return FFEMTfliteEmbedder(
            model_path=ffem_model,
            device='cpu',
            input_shape=input_shape,
            use_normalization=True
        )
    elif lower.endswith('.keras'):
        return FFEMKerasEmbedder(
            model_path=ffem_model,
            device='cpu',
            input_shape=input_shape,
            use_normalization=True
        )
    else:
        raise ValueError(f"Unsupported model format: {ffem_model}")


def compute_embeddings_json(entries, ffem_model, json_path, input_shape=(112,112), use_tta=False, batch_size=1):
    # create cache file path (TTA included)
    model_hash = hash_file(ffem_model)
    json_dir = os.path.dirname(json_path)
    json_base = os.path.splitext(os.path.basename(json_path))[0]
    tta_suffix = "_tta" if use_tta else ""
    cache_path = os.path.join(json_dir, f"{json_base}_{model_hash}{tta_suffix}.npy")

    # check cache file
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        cache_data = np.load(cache_path, allow_pickle=True).item()
        return cache_data['embeddings'], cache_data['labels']

    # calculate embeddings
    embedder = _select_embedder(ffem_model, input_shape)

    if use_tta:
        print("🔄 Using TTA (Test Time Augmentation) with horizontal flip")

    embs, labels = [], []
    if batch_size > 1 and hasattr(embedder, 'get_embeddings_batch'):
        batch_faces = []
        batch_labels = []
        for img_path, info in tqdm(entries, desc="JSON embeddings (batched)"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            x1, y1 = info['x1'], info['y1']
            x2, y2 = info['x2'], info['y2']
            if x1<0: x1=0
            if y1<0: y1=0
            if x2<0: x2=0
            if y2<0: y2=0
            face = img[y1:y2, x1:x2]
            batch_faces.append(face)
            batch_labels.append(os.path.basename(os.path.dirname(img_path)))
            if use_tta:
                batch_faces.append(cv2.flip(face, 1))
                batch_labels.append(os.path.basename(os.path.dirname(img_path)))
            if len(batch_faces) >= batch_size:
                outs, _ = embedder.get_embeddings_batch(batch_faces)
                outs = outs / np.linalg.norm(outs, axis=1, keepdims=True)
                embs.extend(list(outs))
                labels.extend(batch_labels)
                batch_faces, batch_labels = [], []
        if batch_faces:
            outs, _ = embedder.get_embeddings_batch(batch_faces)
            outs = outs / np.linalg.norm(outs, axis=1, keepdims=True)
            embs.extend(list(outs))
            labels.extend(batch_labels)
    else:
        for img_path, info in tqdm(entries, desc="JSON embeddings"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            x1, y1 = info['x1'], info['y1']
            x2, y2 = info['x2'], info['y2']
            if x1<0: x1=0
            if y1<0: y1=0
            if x2<0: x2=0
            if y2<0: y2=0
            face = cv2.resize(img[y1:y2, x1:x2], input_shape)
            e, _ = embedder.get_embedding(face)
            vec = e[0] / np.linalg.norm(e[0])
            embs.append(vec)
            if use_tta:
                face_flipped = cv2.resize(cv2.flip(img[y1:y2, x1:x2], 1), input_shape)
                e_flip, _ = embedder.get_embedding(face_flipped)
                vec_flip = e_flip[0] / np.linalg.norm(e_flip[0])
                embs.append(vec_flip)
            labels.append(os.path.basename(os.path.dirname(img_path)))

    embs = np.vstack(embs) if embs else np.empty((0, input_shape[0]))

    # save cache
    np.save(cache_path, {'embeddings': embs, 'labels': labels})
    print(f"Saved embeddings to {cache_path}")

    return embs, labels

# ─────────── folder-based embedding ────────────

def load_folder_entries(root):
    entries = []
    for identity in os.listdir(root):
        identity_path = os.path.join(root, identity)
        if not os.path.isdir(identity_path):
            continue
        for img_name in os.listdir(identity_path):
            img_path = os.path.join(identity_path, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                entries.append((img_path, identity))
    return entries

def compute_embeddings_folder(entries, ffem_model, input_shape=(112,112), batch_size=1):
    embedder = _select_embedder(ffem_model, input_shape)
    embs, labels = [], []
    if batch_size > 1 and hasattr(embedder, 'get_embeddings_batch'):
        batch_faces = []
        batch_labels = []
        for img_path, identity in tqdm(entries, desc="Folder embeds (batched)"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            batch_faces.append(img)
            batch_labels.append(identity)
            if len(batch_faces) >= batch_size:
                outs, _ = embedder.get_embeddings_batch(batch_faces)
                outs = outs / np.linalg.norm(outs, axis=1, keepdims=True)
                embs.extend(list(outs))
                labels.extend(batch_labels)
                batch_faces, batch_labels = [], []
        if batch_faces:
            outs, _ = embedder.get_embeddings_batch(batch_faces)
            outs = outs / np.linalg.norm(outs, axis=1, keepdims=True)
            embs.extend(list(outs))
            labels.extend(batch_labels)
    else:
        for img_path, identity in tqdm(entries, desc="Folder embeds"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            face = cv2.resize(img, input_shape)
            e, _ = embedder.get_embedding(face)
            vec = e[0] / np.linalg.norm(e[0])
            embs.append(vec)
            labels.append(identity)
    return np.vstack(embs) if embs else np.empty((0, input_shape[0])), labels

# ─────────── pipeline-based embedding ────────────

def create_attrnet_interpreter(attr_model_path):
    interp = fid.create_interpreter(model_path=attr_model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    lm_idx = None
    for od in interp.get_output_details():
        if 'partitionedcall:2' in od['name'].lower():
            lm_idx = od['index']
            break
    if lm_idx is None:
        raise RuntimeError("Cannot find landmark head in AttrNet")
    return interp, inp_det, lm_idx

def compute_embeddings_pipeline(
    entries, yoloface, interp, inp_det, lm_idx,
    ffem_model, attr_input_size=(128,128), ffem_shape=(112,112)
):
    ffem = _select_embedder(ffem_model, ffem_shape)
    embs, labels = [], []
    for img_path, _ in tqdm(entries, desc="Pipeline embeddings"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = fid.detect_yolofaces(
            img,
            yoloface,
            conf_threshold=0.2, max_detections=1, input_size=640
        )
        if len(dets) != 1:
            continue
        x1, y1, x2, y2 = map(int, dets[0]['box'])
        crop = img[y1:y2, x1:x2]

        lb, sc, dx, dy = fid.letterbox(crop, (attr_input_size[0],)*2)
        inp = (
            (lb.astype(np.float32)/255.0 - np.array([0.485,0.456,0.406])) /
            np.array([0.229,0.224,0.225])
        )[None,...].astype(np.float32)
        interp.set_tensor(inp_det['index'], inp)
        interp.invoke()
        lm = interp.get_tensor(lm_idx)[0]

        pts = [
            fid.decode_landmark(
                lm[2*i], lm[2*i+1], x1, y1, sc, dx, dy, attr_input_size[0]
            )
            for i in range(len(lm)//2)
        ]

        yaw, pitch, roll = fid.estimate_pose(
            img, [pts[i] for i in (2,0,1,4,6,7)]
        )
        if abs(yaw)>=30 or abs(pitch)>=30: continue

        aligned = fid.align_face_ffem_bbox(
            img,
            [pts[i] for i in (0,1,2,3,5)],
            -roll,
            output_size=ffem_shape
        )
        if aligned is None:
            continue
        e, _ = ffem.get_embedding(aligned)
        vec = e[0] / np.linalg.norm(e[0])
        embs.append(vec)
        labels.append(os.path.basename(os.path.dirname(img_path)))

    if not embs:
        return np.empty((0,ffem_shape[0])), []
    return np.vstack(embs), labels

# ─────────── distance and ROC calculation ────────────

def pairwise_distances(embs, labels, sample_ratio=1.0, use_sampling=True):
    # Cosine similarity (after L2 normalization): [-1, 1] → map to [0, 1]
    def cos_to_sim(c):
        return 0.5 * (np.clip(c, -1.0, 1.0) + 1.0)

    id2idx = {}
    for i, l in enumerate(labels):
        id2idx.setdefault(l, []).append(i)
    
    # Genuine scores (cosine similarity in [0, 1])
    genuine_scores = []
    for idxs in id2idx.values():
        for i, j in combinations(idxs, 2):
            cos = np.dot(embs[i], embs[j])
            sim = cos_to_sim(cos)
            genuine_scores.append(sim)
    genuine_scores = np.array(genuine_scores)
    
    # Impostor scores (cosine similarity in [0, 1])
    keys = list(id2idx.keys())
    impostor_scores = []
    if use_sampling:
        num_samples = int(len(genuine_scores) * sample_ratio)
        for k in tqdm(range(num_samples), desc="Impostor sampling"):
            a, b = np.random.choice(keys, 2, replace=False)
            i = np.random.choice(id2idx[a])
            j = np.random.choice(id2idx[b])
            cos = np.dot(embs[i], embs[j])
            sim = cos_to_sim(cos)
            impostor_scores.append(sim)
    else:
        # calculate total impostor pairs
        total_pairs = sum(len(id2idx[a]) * len(id2idx[b]) for a, b in combinations(keys, 2))
        with tqdm(total=total_pairs, desc="Impostor exhaustive") as pbar:
            for a, b in combinations(keys, 2):
                for i in id2idx[a]:
                    for j in id2idx[b]:
                        cos = np.dot(embs[i], embs[j])
                        sim = cos_to_sim(cos)
                        impostor_scores.append(sim)
                        pbar.update(1)
    impostor_scores = np.array(impostor_scores)
    
    return genuine_scores, impostor_scores

def analyze(genuine_scores, impostor_scores, bins=50, out_hist="hist.png", out_roc="roc.png", zoom_range=0.05):
    print(f"Genuine: n={len(genuine_scores)} mean={genuine_scores.mean():.4f} med={np.median(genuine_scores):.4f}")
    print(f"Impostor: n={len(impostor_scores)} mean={impostor_scores.mean():.4f} med={np.median(impostor_scores):.4f}")

    plt.figure(figsize=(6,4))
    plt.hist(genuine_scores, bins=bins, alpha=0.6, label='genuine')
    plt.hist(impostor_scores, bins=bins, alpha=0.6, label='impostor')
    plt.legend(); plt.xlabel("Cosine similarity (0-1)"); plt.ylabel("Count")
    plt.savefig(out_hist); plt.close()

    # ROC with similarity as score (higher = more genuine)
    y_true  = np.concatenate([np.ones(len(genuine_scores)),  np.zeros(len(impostor_scores))])
    y_score = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, th  = roc_curve(y_true, y_score)
    auc_sc       = auc(fpr, tpr)
    eer_idx      = np.nanargmin(np.abs(fpr - (1-tpr)))
    eer_thr      = th[eer_idx]  # similarity threshold
    eer_value    = fpr[eer_idx] * 100  # EER as percentage
    vr_eer       = 100 * (np.mean(genuine_scores > eer_thr))  # VR at EER (sim > thr)
    print(f"AUC={auc_sc:.4f}  EER_thr(sim)={eer_thr:.4f}  "
          f"FRR={(1 - np.mean(genuine_scores > eer_thr))*100:.2f}%  "
          f"FAR={np.mean(impostor_scores > eer_thr)*100:.2f}%  "
          f"EER point VR={vr_eer:.2f}%")

    # overall ROC plot
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"AUC={auc_sc:.3f}")
    plt.plot([0,1],[1,0],'--',color='red')  # EER threshold line
    plt.scatter(fpr[eer_idx], tpr[eer_idx], c='red', marker='o', label='EER point')
    plt.text(fpr[eer_idx]+0.02, tpr[eer_idx]-0.02, 
             f'EER={eer_value:.2f}%\nThr(sim)={eer_thr:.3f}\nVR={vr_eer:.2f}%', 
             fontsize=9, verticalalignment='top')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.savefig(out_roc); plt.close()

    # 확대 ROC 플롯
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"AUC={auc_sc:.3f}")
    plt.plot([0,1],[1,0],'--',color='red')  # EER threshold line
    plt.scatter(fpr[eer_idx], tpr[eer_idx], c='red', marker='o', label='EER point')
    plt.text(fpr[eer_idx]+0.005, tpr[eer_idx]-0.005, 
             f'EER={eer_value:.2f}%\nThr(sim)={eer_thr:.3f}\nVR={vr_eer:.2f}%', 
             fontsize=9, verticalalignment='top')
    plt.xlim(max(0, fpr[eer_idx] - zoom_range), fpr[eer_idx] + zoom_range)
    plt.ylim(max(0, tpr[eer_idx] - zoom_range), min(1, tpr[eer_idx] + zoom_range))
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.savefig(out_roc.replace(".png", "_zoomed.png")); plt.close()

    for target in [0.01, 0.02, 0.05]:
        idxs = np.where(fpr <= target)[0]
        if len(idxs)==0: 
            continue
        thr = th[idxs[-1]]  # similarity threshold
        vr  = 100 * (np.mean(genuine_scores > thr))
        print(f"FAR<={int(target*100)}% thr(sim)={thr:.3f} VR={vr:.2f}%")

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="JSON vs Folder vs Pipeline embedding comparison"
    )
    p.add_argument("--root",       required=True, help="image root folder")
    p.add_argument("--ffem_model", required=True, help="FFEM TFLite model(.tflite)")
    p.add_argument("--attr_model", required=True, help="AttrNet TFLite model(.tflite)")
    p.add_argument("--attr_input_size", type=int, default=128)
    p.add_argument("--yolo_face",  default="yolov8n-face.pt")
    p.add_argument("--sample_ratio", type=float, default=1.0, help="Impostor sampling ratio (genuine count based, use_sampling=True when True)")
    p.add_argument("--use_sampling", action='store_true', help="Impostor sampling use (exhaustive calculation when False)")
    p.add_argument("--use_tta", action='store_true', help="Test Time Augmentation use (horizontal flip average when True)")
    p.add_argument("--use_folder", action='store_true', help="Folder-based embedding (ignore JSON when True, assume cropped face images when True)")
    p.add_argument("--json",       default="", help="TrillionPairs bbox JSON (required when use_folder=False)")
    p.add_argument("--zoom_range", type=float, default=0.05, help="ROC zoom range (EER ± zoom_range)")
    p.add_argument("--batch_size", type=int, default=1, help="Embedding calculation batch size (valid for Keras models)")
    args = p.parse_args()

    # validate input
    if not args.use_folder and not args.json:
        raise ValueError("Either --json must be provided or --use_folder must be set")

    # JSON or folder-based embedding
    if args.use_folder:
        entries = load_folder_entries(args.root)
        embs, lbls = compute_embeddings_folder(entries, args.ffem_model, (112,112), batch_size=args.batch_size)
        mode = "Folder"
        out_hist = "hist_folder.png"
        out_roc = "roc_folder.png"
    else:
        entries = load_json_entries(args.json, args.root)
        embs, lbls = compute_embeddings_json(entries, args.ffem_model, args.json, (112,112), use_tta=args.use_tta, batch_size=args.batch_size)
        mode = "JSON"
        out_hist = "hist_json.png"
        out_roc = "roc_json.png"
    genuine_scores, impostor_scores = pairwise_distances(embs, lbls, args.sample_ratio, args.use_sampling)

    # # pipeline-based embedding
    # yoloface = YOLO(args.yolo_face)
    # interp, inp_det, lm_idx = create_attrnet_interpreter(args.attr_model)
    # embs_p, lbls_p = compute_embeddings_pipeline(
    #     entries, yoloface, interp, inp_det, lm_idx,
    #     args.ffem_model,
    #     attr_input_size=(args.attr_input_size,)*2,
    #     ffem_shape=(112,112)
    # )
    # genuine_scores_p, impostor_scores_p = pairwise_distances(embs_p, lbls_p, args.sample_ratio, args.use_sampling)

    # output results
    print(f"=== {mode}-based ===")
    analyze(genuine_scores, impostor_scores, out_hist=out_hist, out_roc=out_roc, zoom_range=args.zoom_range)
    # print("\n=== Pipeline-based ===")
    # analyze(genuine_scores_p, impostor_scores_p, out_hist="hist_pipe.png", out_roc="roc_pipe.png")
    