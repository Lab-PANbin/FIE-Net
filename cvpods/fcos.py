import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import iou_loss
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances
from cvpods.utils import comm, log_first_n
from losses import HMfocalLoss


########################## GRL ########################################
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


#######################################################################

############################## CA #####################################
class FCOSDiscriminator_CA(nn.Module):
    def __init__(self, num_convs=4, in_channels=256, grad_reverse_lambda=0.02, center_aware_weight=80,
                 center_aware_type='ca_feature', grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_CA, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

        # hyperparameters
        assert center_aware_type == 'ca_loss' or center_aware_type == 'ca_feature'
        self.center_aware_weight = center_aware_weight
        self.center_aware_type = center_aware_type

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain

    def forward(self, feature, target, score_map=None, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        # Generate cneter-aware map
        box_cls_map = score_map["box_cls"].clone().sigmoid()
        centerness_map = score_map["centerness"].clone().sigmoid()

        box_regression_map = score_map["box_regression"].clone().sigmoid()
        box_regression_map_mean = box_regression_map.mean(dim=1, keepdim=True)
        E_map = torch.ones_like(box_regression_map_mean)
        box_regression_map_use = E_map - box_regression_map_mean

        n, c, h, w = box_cls_map.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map = maxpooling(box_cls_map)

        # Normalize the center-aware map
        atten_map = (self.center_aware_weight * box_cls_map * centerness_map * box_regression_map_use).sigmoid()

        # Compute loss
        # Center-aware loss (w/ GRL)
        if self.center_aware_type == 'ca_loss':
            if self.grl_applied_domain == 'both':
                feature = self.grad_reverse(feature)
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = self.grad_reverse(feature)

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            # Computer loss
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn_no_reduce(x, target)
            loss = torch.mean(atten_map * loss)

        # Center-aware feature (w/ GRL)
        elif self.center_aware_type == 'ca_feature':
            if self.grl_applied_domain == 'both':
                feature = self.grad_reverse(atten_map * feature)
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = self.grad_reverse(atten_map * feature)

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x, target)

        return loss


##############################################################

################################### GA ############################
class FCOSDiscriminator(nn.Module):
    def __init__(self, num_convs=4, in_channels=256, grad_reverse_lambda=0.02, grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain

    def forward(self, feature, target, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss


#######################################################################################


def permute_all_cls_and_box_to_N_HWA_K_and_concat(
        box_cls, box_delta, box_center, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls, box_delta, box_center


def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[..., 0]
    pred_top = pred[..., 1]
    pred_right = pred[..., 2]
    pred_bottom = pred[..., 3]

    target_left = target[..., 0]
    target_top = target[..., 1]
    target_right = target[..., 2]
    target_bottom = target[..., 3]

    target_aera = (target_left + target_right).clamp_(min=0) * \
                  (target_top + target_bottom).clamp_(min=0)
    pred_aera = (pred_left + pred_right).clamp_(min=0) * \
                (pred_top + pred_bottom).clamp_(min=0)

    w_intersect = (torch.min(pred_left, target_left) +
                   torch.min(pred_right, target_right)).clamp_(min=0)
    h_intersect = (torch.min(pred_bottom, target_bottom) +
                   torch.min(pred_top, target_top)).clamp_(min=0)

    g_w_intersect = (torch.max(pred_left, target_left) +
                     torch.max(pred_right, target_right)).clamp_(min=0)
    g_h_intersect = (torch.max(pred_bottom, target_bottom) +
                     torch.max(pred_top, target_top)).clamp_(min=0)
    ac_uion = (g_w_intersect * g_h_intersect).clamp_(min=0)

    area_intersect = (w_intersect * h_intersect).clamp_(min=0)
    area_union = target_aera + pred_aera - area_intersect
    eps = torch.finfo(torch.float32).eps

    ious = (area_intersect) / (area_union.clamp(min=eps))
    gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)

    return ious, gious


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.quality_branch = cfg.MODEL.FCOS.QUALITY_BRANCH
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS

        self.vfl_loss = HMfocalLoss(
            use_sigmoid=cfg.MODEL.FCOS.VFL.USE_SIGMOID,
            alpha=cfg.MODEL.FCOS.VFL.ALPHA,
            gamma=cfg.MODEL.FCOS.VFL.GAMMA,
            weight_type=cfg.MODEL.FCOS.VFL.WEIGHT_TYPE,
            loss_weight=cfg.MODEL.FCOS.VFL.LOSS_WEIGHT
        )

        # Inference parameters:
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSHead(cfg, feature_shapes)

        self.FCOSDiscriminator = FCOSDiscriminator()
        self.FCOSDiscriminator_CA = FCOSDiscriminator_CA()

        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.object_sizes_of_interest = cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs, is_teacher=False, get_data=False, sup=False, FFT=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                "WARNING",
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)


        if not FFT:
            if not is_teacher:
                if self.training:
                    score_maps = {
                        "box_cls": box_cls,
                        "box_regression": box_delta,
                        "centerness": box_center
                    }
                    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
                    feature_layers = map_layer_to_index.keys()
                    used_feature_layers = ["P3", "P4", "P5", "P6", "P7"]
                    m = {
                        layer: {
                            map_type:
                                score_maps[map_type][map_layer_to_index[layer]]
                            for map_type in score_maps
                        }
                        for layer in feature_layers
                    }
                    f = {
                        layer: features[map_layer_to_index[layer]]
                        for layer in feature_layers
                    }
                    score_maps = m
                    features = f
                    loss_dis_GA = {}
                    loss_dis_CA = {}
                    USE_DIS_GLOBAL = True
                    USE_DIS_CENTER_AWARE = True
                    source_label = 1.0
                    target_label = 0.0
                    if sup:
                        for layer in used_feature_layers:
                            # detatch score_map
                            for map_type in score_maps[layer]:
                                score_maps[layer][map_type] = score_maps[layer][map_type].detach()
                            if USE_DIS_GLOBAL:
                                loss_dis_GA["loss_adv_%s_ds" % layer] = \
                                    0.05 * self.FCOSDiscriminator(features[layer], source_label, domain='source')
                            if USE_DIS_CENTER_AWARE:
                                loss_dis_CA["loss_adv_%s_CA_ds" % layer] = \
                                    0.5 * self.FCOSDiscriminator_CA(features[layer], source_label, score_maps[layer],
                                                                    domain='source')
                        losses_dis_GA_s = sum(loss for loss in loss_dis_GA.values())
                        losses_dis_CA_s = sum(loss for loss in loss_dis_CA.values())
                    if not sup:
                        for layer in used_feature_layers:
                            # detatch score_map
                            for map_type in score_maps[layer]:
                                score_maps[layer][map_type] = score_maps[layer][map_type].detach()
                            if USE_DIS_GLOBAL:
                                loss_dis_GA["loss_adv_%s_dt" % layer] = \
                                    0.05 * self.FCOSDiscriminator(features[layer], target_label, domain='target')
                            if USE_DIS_CENTER_AWARE:
                                loss_dis_CA["loss_adv_%s_CA_dt" % layer] = \
                                    0.5 * self.FCOSDiscriminator_CA(features[layer], target_label, score_maps[layer],
                                                                    domain='target')
                        losses_dis_GA_t = sum(loss for loss in loss_dis_GA.values())
                        losses_dis_CA_t = sum(loss for loss in loss_dis_CA.values())



        if is_teacher:
            box_xyxy = self.get_box(box_cls, box_delta, box_center, shifts, images)
            return box_cls, box_delta, box_center, box_xyxy
        if self.training:
            if get_data:
                box_xyxy = self.get_box(box_cls, box_delta, box_center, shifts, images)
                return box_cls, box_delta, box_center, box_xyxy, losses_dis_GA_t, losses_dis_CA_t

            gt_classes, gt_shifts_reg_deltas, gt_centerness = self.get_ground_truth(
                shifts, gt_instances)
            losses = self.losses(gt_classes, gt_shifts_reg_deltas, gt_centerness,
                               box_cls, box_delta, box_center)

            if FFT:
                return losses
            if not FFT:
                losses['losses_dis_GA_s'] = losses_dis_GA_s
                losses['losses_dis_CA_s'] = losses_dis_CA_s
                return losses

        else:
            results = self.inference(box_cls, box_delta, box_center, shifts,
                                     images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def get_box(self, box_cls, box_delta, box_center, shifts,
                images):
        assert len(shifts) == len(images)
        results = []
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        for img_idx, shifts_per_image in enumerate(shifts):
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            boxes_all = []
            # Iterate over every feature level
            for box_reg_i, shifts_i in zip(box_reg_per_image, shifts_per_image):
                # predict boxes
                predicted_boxes = self.shift2box_transform.apply_deltas(
                    box_reg_i, shifts_i)
                boxes_all.append(predicted_boxes)
            boxes_all = cat(boxes_all)
            results.append(boxes_all)
        return results
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

    def losses(self, gt_classes, gt_shifts_deltas, gt_centerness,
               pred_class_logits, pred_shift_deltas, pred_quality, FFT=False):
        """
        Args:
            For `gt_classes`, `gt_shifts_deltas` and `gt_centerness` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_centerness`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        # pred_class_logits: num-level * (N, C, H, W) -> (- * C)
        # gt_classes: num_image * point, 4 * 18819
        pred_class_logits, pred_shift_deltas, pred_quality = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat(
                pred_class_logits, pred_shift_deltas, pred_quality,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        # print("num_foreground: ", num_foreground)

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())
        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = comm.all_reduce(num_foreground_centerness) / float(comm.get_world_size())

        pos_inds = foreground_idxs
        pos_pred_shift_deltas = pred_shift_deltas[pos_inds]
        pos_gt_shifts_deltas = gt_shifts_deltas[pos_inds]
        pos_labels = gt_classes[pos_inds]

        gt_ious, _ = compute_ious(pos_pred_shift_deltas, pos_gt_shifts_deltas)
        gt_ious = gt_ious.detach()
        cls_iou_targets = torch.zeros_like(pred_class_logits)
        cls_iou_targets[pos_inds, pos_labels] = gt_ious
        loss_cls = self.vfl_loss(
            pred_class_logits[valid_idxs],
            cls_iou_targets[valid_idxs],
            avg_factor=num_foreground
        )

        if self.quality_branch == 'centerness':
            # regression loss
            loss_box_reg = iou_loss(
                pred_shift_deltas[foreground_idxs],
                gt_shifts_deltas[foreground_idxs],
                gt_centerness[foreground_idxs],
                box_mode="ltrb",
                loss_type=self.iou_loss_type,
                reduction="sum",
            ) / max(1.0, num_targets)

            # centerness loss
            loss_quality = F.binary_cross_entropy_with_logits(
                pred_quality[foreground_idxs],
                gt_centerness[foreground_idxs],
                reduction="sum",
            ) / max(1, num_foreground)

        elif self.quality_branch == 'iou':
            loss_box_reg = iou_loss(
                pred_shift_deltas[foreground_idxs],
                gt_shifts_deltas[foreground_idxs],
                box_mode="ltrb",
                loss_type=self.iou_loss_type,
                reduction="sum",
            ) / max(1.0, num_foreground)

            gt_ious, _ = compute_ious(pred_shift_deltas, gt_shifts_deltas)
            gt_ious = gt_ious.unsqueeze(1).detach()
            loss_quality = F.binary_cross_entropy_with_logits(
                pred_quality[foreground_idxs],
                gt_ious[foreground_idxs],
                reduction="sum",
            ) / max(1, num_foreground)
        else:
            raise Exception(f'Unknown Quality Branch type: {self.quality_branch}')

        if FFT:
            return {
                "loss_cls_fft": loss_cls,
                "loss_box_reg_fft": loss_box_reg,
                "loss_quality_fft": loss_quality
            }

        if not FFT:
            return {
                "loss_cls": loss_cls,
                "loss_box_reg": loss_box_reg,
                "loss_quality": loss_quality
            }

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets):

        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        for shifts_per_image, targets_per_image in zip(shifts, targets):
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.size(0), -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == math.inf] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(
            gt_shifts_deltas), torch.stack(gt_centerness)

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image_box(self, box_cls, box_delta, box_center, shifts,
                                   image_size):
        boxes_all = []

        # Iterate over every feature level
        for box_reg_i, shifts_i in zip(box_delta, shifts):
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)
            boxes_all.append(predicted_boxes)
        boxes_all = cat(boxes_all)
        return boxes_all

    def inference_single_image(self, box_cls, box_delta, box_center, shifts,
                               image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            # (HxWxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = generalized_batched_nms(
            boxes_all, scores_all, class_idxs_all,
            self.nms_threshold, nms_type=self.nms_type
        )
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)

        results = self.inference(box_cls, box_delta, box_center, shifts, images)
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        num_shifts = cfg.build_shift_generator(cfg, input_shape).num_cell_shifts
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # fmt: on
        assert len(set(num_shifts)) == 1, "using differenct num_shifts value is not supported"
        num_shifts = num_shifts[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_shifts * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_shifts * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.centerness = nn.Conv2d(in_channels,
                                    num_shifts * 1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score,
            self.bbox_pred, self.centerness
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits = []
        bbox_reg = []
        centerness = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))

            if self.norm_reg_targets:
                bbox_scaled = F.relu(bbox_pred) * self.fpn_strides[level]
                bbox_reg.append(bbox_scaled)
            else:
                bbox_scaled = torch.exp(bbox_pred)
                bbox_reg.append(bbox_scaled)
        return logits, bbox_reg, centerness
