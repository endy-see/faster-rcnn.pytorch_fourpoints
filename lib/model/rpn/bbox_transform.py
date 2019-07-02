# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets


def quadbox_transform_batch(ex_rois, gt_quadrois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        #ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        #ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_quadx1 = gt_quadrois[:, :, 0]
        gt_quady1 = gt_quadrois[:, :, 1]
        gt_quadx2 = gt_quadrois[:, :, 2]
        gt_quady2 = gt_quadrois[:, :, 3]
        gt_quadx3 = gt_quadrois[:, :, 4]
        gt_quady3 = gt_quadrois[:, :, 5]
        gt_quadx4 = gt_quadrois[:, :, 6]
        gt_quady4 = gt_quadrois[:, :, 7]

        targets_dx1 = (gt_quadx1 - ex_rois[:, 0].view(1,-1).expand_as(gt_quadx1)) / ex_widths
        targets_dy1 = (gt_quady1 - ex_rois[:, 1].view(1,-1).expand_as(gt_quady1)) / ex_heights
        targets_dx2 = (gt_quadx2 - ex_rois[:, 2].view(1,-1).expand_as(gt_quadx2)) / ex_widths
        targets_dy2 = (gt_quady2 - ex_rois[:, 1].view(1,-1).expand_as(gt_quady2)) / ex_heights
        targets_dx3 = (gt_quadx3 - ex_rois[:, 2].view(1, -1).expand_as(gt_quadx3)) / ex_widths
        targets_dy3 = (gt_quady3 - ex_rois[:, 3].view(1, -1).expand_as(gt_quady3)) / ex_heights
        targets_dx4 = (gt_quadx4 - ex_rois[:, 0].view(1, -1).expand_as(gt_quadx4)) / ex_widths
        targets_dy4 = (gt_quady4 - ex_rois[:, 3].view(1, -1).expand_as(gt_quady4)) / ex_heights

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        #ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        #ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_quadx1 = gt_quadrois[:, :, 0]
        gt_quady1 = gt_quadrois[:, :, 1]
        gt_quadx2 = gt_quadrois[:, :, 2]
        gt_quady2 = gt_quadrois[:, :, 3]
        gt_quadx3 = gt_quadrois[:, :, 4]
        gt_quady3 = gt_quadrois[:, :, 5]
        gt_quadx4 = gt_quadrois[:, :, 6]
        gt_quady4 = gt_quadrois[:, :, 7]

        targets_dx1 = (gt_quadx1 - ex_rois[:, :, 0]) / ex_widths
        targets_dy1 = (gt_quady1 - ex_rois[:, :, 1]) / ex_heights
        targets_dx2 = (gt_quadx2 - ex_rois[:, :, 2]) / ex_widths
        targets_dy2 = (gt_quady2 - ex_rois[:, :, 1]) / ex_heights
        targets_dx3 = (gt_quadx3 - ex_rois[:, :, 2]) / ex_widths
        targets_dy3 = (gt_quady3 - ex_rois[:, :, 3]) / ex_heights
        targets_dx4 = (gt_quadx4 - ex_rois[:, :, 0]) / ex_widths
        targets_dy4 = (gt_quady4 - ex_rois[:, :, 3]) / ex_heights
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx1, targets_dy1, targets_dx2, targets_dy2,
         targets_dx3, targets_dy3, targets_dx4, targets_dy4),2)

    return targets

def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def quadbox_transform_inv(boxes, quaddeltas):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    #ctr_x = boxes[:, :, 0] + 0.5 * widths
    #ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx1 = quaddeltas[:, :, 0::8]
    dy1 = quaddeltas[:, :, 1::8]
    dx2 = quaddeltas[:, :, 2::8]
    dy2 = quaddeltas[:, :, 3::8]
    dx3 = quaddeltas[:, :, 4::8]
    dy3 = quaddeltas[:, :, 5::8]
    dx4 = quaddeltas[:, :, 6::8]
    dy4 = quaddeltas[:, :, 7::8]

    pred_quadx1 = dx1 * widths.unsqueeze(2) + boxes[:, :, 0].unsqueeze(2)
    pred_quady1 = dy1 * heights.unsqueeze(2) + boxes[:, :, 1].unsqueeze(2)
    pred_quadx2 = dx2 * widths.unsqueeze(2) + boxes[:, :, 2].unsqueeze(2)
    pred_quady2 = dy2 * heights.unsqueeze(2) + boxes[:, :, 1].unsqueeze(2)
    pred_quadx3 = dx3 * widths.unsqueeze(2) + boxes[:, :, 2].unsqueeze(2)
    pred_quady3 = dy3 * heights.unsqueeze(2) + boxes[:, :, 3].unsqueeze(2)
    pred_quadx4 = dx4 * widths.unsqueeze(2) + boxes[:, :, 0].unsqueeze(2)
    pred_quady4 = dy4 * heights.unsqueeze(2) + boxes[:, :, 3].unsqueeze(2)

    pred_quadboxes = quaddeltas.clone()
    pred_quadboxes[:, :, 0::8] = pred_quadx1
    pred_quadboxes[:, :, 1::8] = pred_quady1
    pred_quadboxes[:, :, 2::8] = pred_quadx2
    pred_quadboxes[:, :, 3::8] = pred_quady2
    pred_quadboxes[:, :, 4::8] = pred_quadx3
    pred_quadboxes[:, :, 5::8] = pred_quady3
    pred_quadboxes[:, :, 6::8] = pred_quadx4
    pred_quadboxes[:, :, 7::8] = pred_quady4

    return pred_quadboxes


def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes


def clip_quadboxes(quadboxes, im_shape, batch_size):

    for i in range(batch_size):
        quadboxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        quadboxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        quadboxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        quadboxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)
        quadboxes[i,:,4::4].clamp_(0, im_shape[i, 1]-1)
        quadboxes[i,:,5::4].clamp_(0, im_shape[i, 0]-1)
        quadboxes[i,:,6::4].clamp_(0, im_shape[i, 1]-1)
        quadboxes[i,:,7::4].clamp_(0, im_shape[i, 0]-1)
    return quadboxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps
