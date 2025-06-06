import os
import json

images_dir = "../data_packs/Clouds-1000/all_images_and_labels/images"
meta_path = "../data_packs/Clouds-1000/meta.json"
output_path = "../data_packs/Clouds-1000/image_labels.json"

def generate_labels(label="Cumuliformes"):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    all_classes = [c['title'] for c in meta['classes']]

    if label not in all_classes:
        raise ValueError(f"Label '{label}' not found in meta.json")

    labels = {}
    for fname in os.listdir(images_dir):
        if fname.lower().endswith('.jpg'):
            labels[fname] = label

    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {len(labels)} labels to {output_path}")

if __name__ == "__main__":
    generate_labels("Cumuliformes")
