"""Predict one or more local images without starting the UI."""
import argparse
from common import load_classifier, image_batch, local_path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', nargs='+')
    p.add_argument('--model')
    p.add_argument('--labels')
    args = p.parse_args()
    from PIL import Image
    import numpy as np
    model, labels = load_classifier(args.model, args.labels)
    for path in args.images:
        with Image.open(local_path(path)) as img:
            scores = model(image_batch(img), training=False).numpy()[0]
        print(path)
        for idx in np.argsort(scores)[::-1][:3]:
            print(f'  {labels[idx]}: {scores[idx]*100:.2f}%')

if __name__ == '__main__':
    main()
