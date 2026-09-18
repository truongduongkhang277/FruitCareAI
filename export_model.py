"""Optional H5/TFLite export; keeps training independent of converter errors."""
import argparse
from common import load_classifier, local_path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model')
    p.add_argument('--labels')
    p.add_argument('--format', choices=['h5','tflite'], default='tflite')
    p.add_argument('--output', default='models')
    args = p.parse_args()
    import tensorflow as tf
    import numpy as np
    model, _ = load_classifier(args.model, args.labels)
    out = local_path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    path = out / ('fruit_classifier_model.' + args.format)
    if args.format == 'h5':
        model.save(path)
    else:
        # Freeze an inference-only graph so random augmentation is not exported.
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        @tf.function(input_signature=[tf.TensorSpec([1,224,224,3], tf.float32)])
        def serving(x):
            return model(x, training=False)
        frozen = convert_variables_to_constants_v2(serving.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        data = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=data)
        interpreter.allocate_tensors()
        sample = np.random.default_rng(42).uniform(0,255,(1,224,224,3)).astype('float32')
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],sample)
        interpreter.invoke()
        actual = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        expected = model(sample,training=False).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
        path.write_bytes(data)
    print(f'Đã xuất: {path}')

if __name__ == '__main__':
    main()
