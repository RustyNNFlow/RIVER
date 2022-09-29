use crate::{
    nn,
    Tensor,
    Reduction,
    kind,
    Kind,
};
use serde::{Serialize, Deserialize};

pub fn bbox_overlaps(
    bboxes1:Tensor,
    bboxes2:Tensor,
)->Tensor{

    let rows = bboxes1.size()[0];
    let cols = bboxes2.size()[0];
    // bboxes1.print();
    // bboxes2.print();
    assert_eq!(rows, cols);
    if rows * cols == 0{
        return Tensor::zeros(&[rows,1], kind::FLOAT_CPU);
    }

    let (l, _) =  Tensor::cat(&[bboxes1.narrow(1,0,1), bboxes2.narrow(1,0,1)],1).max_dim(-1, true);
    let (t, _) =  Tensor::cat(&[bboxes1.narrow(1,1,1), bboxes2.narrow(1,1,1)],1).max_dim(-1, true);
    let (r, _) =  Tensor::cat(&[bboxes1.narrow(1,2,1), bboxes2.narrow(1,2,1)],1).min_dim(-1, true);
    let (b, _) =  Tensor::cat(&[bboxes1.narrow(1,3,1), bboxes2.narrow(1,3,1)],1).min_dim(-1, true);
    let w = (r-l+1).clamp_min(0);
    let h = (b-t+1).clamp_min(0);
    let overlap = w*h;
    let area1 = (bboxes1.narrow(1,2,1)-bboxes1.narrow(1,0,1)+1)*(bboxes1.narrow(1,3,1)-bboxes1.narrow(1,1,1)+1);
    let area2 = (bboxes2.narrow(1,2,1)-bboxes2.narrow(1,0,1)+1)*(bboxes2.narrow(1,3,1)-bboxes2.narrow(1,1,1)+1);
    let ious = overlap.copy()/(area1+area2-overlap);

    return ious;
}
pub fn iou_loss(
    pred:Tensor,
    target:Tensor,
    eps:f64,
)->Tensor{
    let ious = bbox_overlaps(pred, target).clamp_min(eps);
    -ious.log()
}
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct IoULossCfg{
    eps:f64,
    loss_weight:f64,
}

impl IoULossCfg {
    pub fn loads(json_str: &String) -> IoULossCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct IoULoss{
    eps:f64,
    loss_weight:f64,
}

impl IoULoss {
    pub fn new(
        cfg: &IoULossCfg,
    )->IoULoss{
        IoULoss{
            eps:cfg.eps,
            loss_weight:cfg.loss_weight,
        }
    }
    pub fn forward(
        &self,
        pred:Tensor,
        target:Tensor,
        avg_factor:i64,
    ) -> Tensor {
        self.loss_weight*iou_loss(pred, target, self.eps).sum(Kind::Double)/avg_factor
    }
}



