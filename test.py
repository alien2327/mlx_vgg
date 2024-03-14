import sys

from vgg.utils import load_image
from vgg.utils import decode_probabilities

from vgg.vgg16 import VGG16
from vgg.vgg19 import VGG19

test_image = load_image(sys.argv[1])

weight_path = 'resources/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
model = VGG19().load_weights(weight_path)
pred_prob = model(test_image, train=False, out_type='np')

label = decode_probabilities(pred_prob[0])[0]
print(f"Predicted: {label[0]} ({label[1]*100:.2f}%)")
