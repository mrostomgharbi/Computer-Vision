from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path")

parser.add_option("-n", "--num_rois", type="int", dest="num_rois",default=32) 

parser.add_option("--network", dest="network", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", default=2000)
parser.add_option("--config_filename", dest="config_filename", default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path")
(options, args) = parser.parse_args()

from keras_frcnn.pascal_voc_parser import get_data

C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
else:
    print('Not a valid model')


if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    C.base_net_weights = nn.get_weight_path()

images_, count_c, c_map = get_data(options.train_path)

if 'bg' not in count_c:
    count_c['bg'] = 0
    c_map['bg'] = len(c_map)

C.c_map = c_map

inv_map = {v: k for k, v in c_map.items()}


config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)

random.shuffle(images_)

num_imgs = len(images_)

train_ = [s for s in images_ if s['imageset'] == 'trainval']
val_ = [s for s in images_ if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_)))
print('Num val samples {}'.format(len(val_)))


data_gen_train = data_generators.get_anchor_gt(train_, count_c, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_, count_c, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

imgi = Input(shape=input_shape_img)
roii = Input(shape=(None, 4))

# define the base network (resnet,VGG, Inception, etc)
shared_layers = nn.nn_base(imgi, trainable=True)

# define the RPN : 
nbr_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, nbr_anchors)

classifier = nn.classifier(shared_layers, roii, C.num_rois, nb_classes=len(count_c), trainable=True)

model_rpn = Model(imgi, rpn[:2])
model_classifier = Model([imgi, roii], classifier)

model_all = Model([imgi, roii], rpn[:2] + classifier)

optimizer = Adam(lr=1e-5)
optimizer_clf = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(nbr_anchors), losses.rpn_loss_regr(nbr_anchors)])
model_classifier.compile(optimizer=optimizer_clf, loss=[losses.class_loss_cls, losses.class_loss_regr(len(count_c)-1)], metrics={'dense_class_{}'.format(len(count_c)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 100
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

c_map_inv = {v: k for k, v in c_map.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:

            if len(rpn_accuracy_monitor) == epoch_length and C.verbose:
                mob = float(sum(rpn_accuracy_monitor))/len(rpn_accuracy_monitor)
                rpn_accuracy_monitor = []

            X, Y, img_data = next(data_gen_train)

            l_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, c_map)

            if X2 is None:
                rpn_accuracy_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_s = np.where(Y1[0, :, -1] == 1)
            pos_s = np.where(Y1[0, :, -1] == 0)

            if len(neg_s) > 0:
                neg_s = neg_s[0]
            else:
                neg_s = []

            if len(pos_s) > 0:
                pos_s = pos_s[0]
            else:
                pos_s = []

            rpn_accuracy_monitor.append(len(pos_s))
            rpn_accuracy_for_epoch.append((len(pos_s)))

            if C.num_rois > 1:
                if len(pos_s) < C.num_rois//2:
                    select_pos_s = pos_s.tolist()
                else:
                    select_pos_s = np.random.choice(pos_s, C.num_rois//2, replace=False).tolist()
                try:
                    selected_neg_s = np.random.choice(neg_s, C.num_rois - len(select_pos_s), replace=False).tolist()
                except:
                    selected_neg_s = np.random.choice(neg_s, C.num_rois - len(select_pos_s), replace=True).tolist()

                sel_samples = select_pos_s + selected_neg_s
            else:
                select_pos_s = pos_s.tolist()
                selected_neg_s = neg_s.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_s)
                else:
                    sel_samples = random.choice(pos_s)

            l_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = l_rpn[1]
            losses[iter_num, 1] = l_rpn[2]

            losses[iter_num, 2] = l_class[1]
            losses[iter_num, 3] = l_class[2]
            losses[iter_num, 4] = l_class[3]

            progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                      ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

            iter_num += 1

            if iter_num == epoch_length:
                l_rpn_cls = np.mean(losses[:, 0])
                l_rpn_regr = np.mean(losses[:, 1])
                l_class_cls = np.mean(losses[:, 2])
                l_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mob = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mob))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(l_rpn_cls))
                    print('Loss RPN regression: {}'.format(l_rpn_regr))
                    print('Loss Detector classifier: {}'.format(l_class_cls))
                    print('Loss Detector regression: {}'.format(l_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = l_rpn_cls + l_rpn_regr + l_class_cls + l_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                break

        except Exception as e:
            continue

print('Training complete, exiting.')
