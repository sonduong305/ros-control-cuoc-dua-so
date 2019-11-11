from tensorflow.contrib import slim
# from tf_bisenet.frontends import resnet_v2
# from tf_bisenet.frontends import mobilenet_v2
# from tf_bisenet.frontends import inception_v4
# from tf_bisenet.frontends import densenet
from tf_bisenet.frontends import xception
import os 
import subprocess


def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_frontend(inputs, frontend_config, is_training=True, reuse=False):
    frontend = frontend_config['frontend']
    pretrained_dir = frontend_config['pretrained_dir']

    if "ResNet50" == frontend and not os.path.isfile("pretrain/resnet_v2_50.ckpt"):
        download_checkpoints("ResNet50")
    if "ResNet101" == frontend and not os.path.isfile("pretrain/resnet_v2_101.ckpt"):
        download_checkpoints("ResNet101")
    if "ResNet152" == frontend and not os.path.isfile("pretrain/resnet_v2_152.ckpt"):
        download_checkpoints("ResNet152")
    if "MobileNetV2" == frontend and not os.path.isfile("pretrain/mobilenet_v2.ckpt.data-00000-of-00001"):
        download_checkpoints("MobileNetV2")
    if "InceptionV4" == frontend and not os.path.isfile("pretrain/inception_v4.ckpt"):
        download_checkpoints("InceptionV4")

    if frontend == 'Xception39':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception39(inputs, is_training=is_training, scope='xception39', reuse=reuse)
            frontend_scope='Xception39'
            init_fn = None
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))

    return logits, end_points, frontend_scope, init_fn 