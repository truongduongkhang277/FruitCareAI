"""Evaluate on held-out test and measure real inference time."""
import argparse
import json
import time
from common import local_path, load_classifier

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='fruit_dataset/dataset/test')
    p.add_argument('--model')
    p.add_argument('--labels')
    p.add_argument('--output', default='outputs')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    data = local_path(args.data)
    if not data.is_dir() or args.batch_size < 1:
        p.error('Kiểm tra thư mục test và batch-size > 0.')
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
    model, labels = load_classifier(args.model, args.labels)
    ds = tf.keras.utils.image_dataset_from_directory(str(data), class_names=labels,
        image_size=(224,224), batch_size=args.batch_size, shuffle=False, interpolation='bilinear')
    y_true, probabilities = [], []
    for images, target in ds:
        probabilities.extend(model(images, training=False).numpy())
        y_true.extend(target.numpy())
    probabilities = np.asarray(probabilities)
    y_pred = np.argmax(probabilities, axis=1)
    indices = list(range(len(labels)))
    report = classification_report(y_true, y_pred, labels=indices, target_names=labels, output_dict=True, zero_division=0)
    sample = next(iter(ds))[0][:1]
    for _ in range(3):
        model(sample, training=False).numpy()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        model(sample, training=False).numpy()
        times.append((time.perf_counter()-start)*1000)
    metrics = dict(test_accuracy=float(accuracy_score(y_true,y_pred)),
                   test_loss=float(log_loss(y_true, probabilities, labels=indices)),
                   inference_mean_ms=float(np.mean(times)), inference_p95_ms=float(np.percentile(times,95)),
                   timing_note='Batch 1, 3 warmups, 20 runs; excludes file decoding/resizing',
                   test_images=len(y_true), model=str(args.model or 'from .env/default'),
                   classification_report=report)
    out = local_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(classification_report(y_true, y_pred, labels=indices, target_names=labels, zero_division=0))
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}; loss: {metrics['test_loss']:.4f}")
    print(f"Inference mean: {metrics['inference_mean_ms']:.2f} ms/ảnh")
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(confusion_matrix(y_true,y_pred,labels=indices), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
    ax.set(xlabel='Predicted label', ylabel='True label', title='FruitCareAI — held-out test')
    fig.tight_layout()
    fig.savefig(out/'confusion_matrix.png', dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    main()
