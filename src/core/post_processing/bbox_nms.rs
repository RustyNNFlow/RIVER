use crate::{
    nn,
    Tensor,
    kind,
};
use std::collections::HashSet;

pub fn multiclass_nms(
    multi_bboxes: &Tensor,
    multi_scores: &Tensor,
    score_thr: f64,
    iou_thr:f64,
)->(Vec<Tensor>, Vec<Tensor>)
{
    // multi_bboxes (Tensor): shape (n, 4)
    // multi_scores (Tensor): shape (n, #class), where the 0th column
    // contains scores of the background class, but this will be ignored.
    let num_classes = multi_scores.size()[1];
    let num_ = multi_scores.size()[0];
    let mut bboxes:Vec<Tensor> = Vec::new();
    let mut labels:Vec<Tensor> = Vec::new();
    for i in 1..num_classes{
        let cls_inds = multi_scores.select(1, i).gt(score_thr);
        if 0 == cls_inds.any().reshape(&[1]).int64_value(&[0]){
            continue;
        }
        let cls_inds_tmp = cls_inds.reshape(&[num_, 1]).expand(&[num_, 4], false);
        let bboxes_tmp = multi_bboxes
            .masked_select(&cls_inds_tmp)
            .reshape(&[-1, 4]);

        let scores_tmp = multi_scores
            .select(1, i)
            .masked_select(&cls_inds)
            .reshape(&[-1, 1]);
        // println!("{:?} {:?}",bboxes_tmp, scores_tmp);
        let (bboxes_tmp, scores_tmp)= nms_op(&bboxes_tmp, &scores_tmp, iou_thr);
        // println!("{:?} {:?}",bboxes_tmp, scores_tmp);
        let cls_dets = Tensor::cat(&[bboxes_tmp, scores_tmp], 1);
        // let cls_labels = multi_bboxes.new_full(&[cls_dets.size()[0]], i-1);
        let cls_labels = Tensor::zeros(&[cls_dets.size()[0]], kind::INT64_CPU).g_add_scalar(i);//.to_device(vs.device());
        bboxes.push(cls_dets);
        labels.push(cls_labels)
    }
    (bboxes, labels)
}
pub fn xmin(t:&Tensor)->Tensor{
    t.select(0,0)
}
pub fn ymin(t:&Tensor)->Tensor{
    t.select(0,1)
}
pub fn xmax(t:&Tensor)->Tensor{
    t.select(0,2)
}
pub fn ymax(t:&Tensor)->Tensor{
    t.select(0,3)
}
pub fn area(t:&Tensor)->Tensor{
    (xmax(t)-xmin(t)+1.)*(ymax(t)-ymin(t)+1.)
}

// // Intersection over union of two bounding boxes.
pub fn iou(b1: &Tensor, b2: &Tensor) -> Tensor {
    let b1_area = area(&b1);
    let b2_area = area(&b2);
    let i_xmin = xmin(&b1).fmax(&xmin(&b2));
    let i_xmax = xmax(&b1).fmin(&xmax(&b2));
    let i_ymin = ymin(&b1).fmax(&ymin(&b2));
    let i_ymax = ymax(&b1).fmin(&ymax(&b2));
    let i_area = (i_xmax - i_xmin + 1.).clamp_min(0) * (i_ymax - i_ymin + 1.).clamp_min(0);
    i_area.g_div(&b1_area.g_add(&b2_area).g_sub(&i_area))
}

pub fn nms_op(
    bboxes:&Tensor,
    scores:&Tensor,
    iou_thr:f64,
)->(Tensor, Tensor){
    let (sorted_scores, sort_index) = scores.sort(0, true);
    let sorted_bboxes = bboxes.index_select(0, &sort_index.reshape(&[-1]));
    let num_ = sorted_scores.size()[0];
    let mut drop_set = HashSet::new();

    // sorted_bboxes.print();
    for index in 0..num_-1{
        if drop_set.contains(&index){
            continue
        }
        for prev_index in index+1..num_{
            if drop_set.contains(&prev_index){
                continue
            }
            // sorted_bboxes.select(0,index).print();
            // sorted_bboxes.select(0, prev_index).print();
            let iou_o = iou(&sorted_bboxes.select(0,index), &sorted_bboxes.select(0, prev_index));
            // iou_o.print();
            if iou_o.reshape(&[1]).double_value(&[0])>iou_thr{
                drop_set.insert(prev_index);
            }
        }
    }
    let mut all_vec:Vec<i64> = Vec::new();
    for i in 0..num_{
        if !drop_set.contains(&i){
            all_vec.push(i);
        }
    }
    let keep_tensor = Tensor::of_slice(&all_vec);
    (sorted_bboxes.index_select(0, &keep_tensor), sorted_scores.index_select(0, &keep_tensor))
}