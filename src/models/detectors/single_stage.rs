use crate::{
    models::backbones::resnet,
    models::necks::bifpn,
    models::bbox_heads::fcos_head,
    nn,
    Tensor,
};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct SingleStageDetectorCfg{
    backbone:resnet::ResNetCfg,
    neck:bifpn::BiFPNCfg,
    bbox_head:fcos_head::FCOSHeadCfg,
}

impl SingleStageDetectorCfg {
    pub fn loads(json_str: &String) -> SingleStageDetectorCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct SingleStageDetector{
    backbone: resnet::ResNet,
    neck: bifpn::BiFPN,
    bbox_head: fcos_head::FCOSHead,
}

impl SingleStageDetector {
    pub fn new(
        p: &nn::Path,
        cfg: &SingleStageDetectorCfg,
    )->SingleStageDetector{
        SingleStageDetector{
            backbone: resnet::ResNet::new(p, &cfg.backbone),
            neck: bifpn::BiFPN::new(p, &cfg.neck),
            bbox_head: fcos_head::FCOSHead::new(p, &cfg.bbox_head),
        }
    }
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Vec<Tensor> {
        let xs = &self.backbone.forward_t(input, train);

        let xs = &self.neck.forward_t(xs, train);

        self.bbox_head.forward_t(xs, train)
    }
    pub fn forward_train(
        &self,
        input: &Tensor,
        gt_bboxes:&Vec<Tensor>,
        gt_labels:&Vec<Tensor>,
        train: bool,
    )-> Tensor
    {
        let xs = self.forward_t(input, train);

        let mut cls_scores:Vec<Tensor>=Vec::new();
        let mut bbox_preds:Vec<Tensor>=Vec::new();
        let level_num = xs.len();
        let class_num = self.bbox_head.num_classes;
        for i in 0..level_num{
            cls_scores.push(xs[i].narrow(1, 0, class_num-1));
            bbox_preds.push(xs[i].narrow(1, class_num-1,4));
        }

        self.bbox_head.loss(
            &cls_scores,
            &bbox_preds,
            gt_bboxes,
            gt_labels,
        )

    }
}
