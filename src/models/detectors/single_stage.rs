use crate::{
    models::backbones::resnet,
    models::necks::bifpn,
    models::bbox_heads::fcos_head,
    nn,
    Tensor,
    Kind,
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
    // pub fn forward_train(
    //     &self,
    //     xs: &Tensor,
    //     ys:&Tensor,
    // )-> Tensor
    // {
    //     nn::ModuleT::forward_t(self, xs, true).cross_entropy_for_logits(&ys)
    // }
    // pub fn simple_test(
    //     &self,
    //     xs: &Tensor,
    // )-> (Tensor, Tensor)
    // {
    //     let scores = nn::ModuleT::forward_t(self, &xs, false).softmax(-1, Kind::Float);
    //     let ids = scores.argmax(-1, false);
    //     (scores, ids)
    // }
}
