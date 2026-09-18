"""Train EfficientNetB0 in two stages; reserve dataset/test for final evaluation."""
import argparse
import json
from common import local_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='fruit_dataset/dataset')
    p.add_argument('--output', default='models', help='Use a separate folder for a new experiment.')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs-head', type=int, default=8)
    p.add_argument('--epochs-fine', type=int, default=15)
    p.add_argument('--validation-split', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.batch_size < 1 or args.epochs_head < 1 or args.epochs_fine < 0:
        p.error('batch-size và epochs-head phải > 0; epochs-fine phải >= 0.')
    if not 0 < args.validation_split < 1:
        p.error('validation-split phải nằm giữa 0 và 1.')
    train_dir = local_path(args.data) / 'train'
    if not train_dir.is_dir():
        p.error(f'Không thấy {train_dir}. Chạy download_data.py hoặc sửa --data.')
    out = local_path(args.output)
    if (out/'fruit_classifier_model.keras').exists():
        p.error('Thư mục đã có model. Chọn --output khác để không ghi đè lần trước.')
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(directory=str(train_dir), image_size=(224,224), batch_size=args.batch_size,
                  label_mode='int', validation_split=args.validation_split, seed=args.seed,
                  shuffle=True, interpolation='bilinear')
    train = tf.keras.utils.image_dataset_from_directory(subset='training', **kwargs)
    val = tf.keras.utils.image_dataset_from_directory(subset='validation', **kwargs)
    labels = train.class_names
    if len(labels) < 2:
        raise ValueError('Cần ít nhất hai thư mục lớp trong train.')
    (out/'fruit_class_names.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
    # Không cache toàn bộ ảnh trong RAM; prefetch giới hạn cho máy cá nhân.
    train = train.prefetch(1)
    val = val.prefetch(1)
    base = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input((224,224,3))
    x = layers.RandomFlip('horizontal_and_vertical')(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    # BN của backbone giữ thống kê ImageNet trong cả hai giai đoạn.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(labels), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name='FruitCareAI')
    def compile_model(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Dùng cùng checkpoint callback để giữ val_loss tốt nhất qua hai giai đoạn.
    best = out/'fruit_classifier_model.keras'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(best), monitor='val_loss', save_best_only=True)
    compile_model(1e-3)
    history = model.fit(train, validation_data=val, epochs=args.epochs_head, callbacks=[checkpoint]).history
    if args.epochs_fine:
        base.trainable = True
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-50:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        compile_model(1e-5)
        h2 = model.fit(train, validation_data=val, epochs=args.epochs_fine, callbacks=[checkpoint]).history
        for key in history:
            history[key] += h2[key]
    (out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (out/'training_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    for ax, metric in zip(axes, ['accuracy','loss']):
        ax.plot(history[metric], label='Train')
        ax.plot(history['val_'+metric], label='Validation')
        ax.axvline(args.epochs_head-0.5, linestyle='--', color='gray', label='Fine-tuning')
        ax.set(title=metric, xlabel='Epoch (0-based)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'training_history.png', dpi=160)
    plt.close(fig)
    print(f'Đã lưu checkpoint có val_loss thấp nhất: {best}')
    print('Tiếp theo: python evaluate.py ; python app.py')

if __name__ == '__main__':
    main()
