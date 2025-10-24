import csv
import json
from trillion_pairs import rect_from_landmark


def save_bbox_to_json(lndmk_file, out_file, method='bbox'):
    """
    create bbox from VGG-Face2 dataset using bbox or landmark
    
    Args:
        lndmk_file: landmark CSV file path
        out_file: output JSON file path
        method: 'bbox' or 'landmarks'
    """
    results = {}
    label_count = -1
    label_book = {}
    image_count = {}
    
    with open(lndmk_file, 'r') as lndmk_f:
        reader = csv.DictReader(lndmk_f, delimiter=',')
        for n, line in enumerate(reader):
            img_path = line['NAME_ID'] + '.jpg'
            label = line['NAME_ID'].split('/')[0]
            
            if label not in label_book:
                label_count += 1
                label_book[label] = label_count
                image_count[label] = 1
            else:
                image_count[label] += 1
            label = label_book[label]
            
            if method == 'bbox':
                # original bbox method (use X, Y, W, H)
                x1 = int(line['X'])
                y1 = int(line['Y'])
                x2 = x1 + int(line['W'])
                y2 = y1 + int(line['H'])
            elif method == 'landmarks':
                # landmark method (use P1X,P1Y ~ P5X,P5Y)
                landmarks_5pt = [
                    float(line['P1X']), float(line['P1Y']),  # left eye
                    float(line['P2X']), float(line['P2Y']),  # right eye
                    float(line['P3X']), float(line['P3Y']),  # nose
                    float(line['P4X']), float(line['P4Y']),  # left mouth
                    float(line['P5X']), float(line['P5Y'])   # right mouth
                ]
                x1, y1, x2, y2 = rect_from_landmark(landmarks_5pt)
                # convert numpy type to Python int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[img_path] = {
                'label': label,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            }
            if n % 10000 == 0:
                print(f'{n} images processed')

    with open(out_file, 'w') as out_f:
        json.dump(results, out_f, indent=2)
