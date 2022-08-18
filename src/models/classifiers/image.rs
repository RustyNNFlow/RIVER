use crate::{
    models::backbones::resnet,
    models::necks::gap,
    models::cls_heads::linear_head,
    nn,
    Tensor,
};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct ImageClassifierCfg{
    backbone:resnet::ResNetCfg,
    neck:gap::GlobalAveragePoolingCfg,
    head:linear_head::LinearClsHeadCfg,
}

impl ImageClassifierCfg {
    pub fn loads(json_str: &String) -> ImageClassifierCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct ImageClassifier{
    backbone: resnet::ResNet,
    neck: gap::GlobalAveragePooling,
    head: linear_head::LinearClsHead,
}

impl ImageClassifier {
    pub fn new(
        p: &nn::Path,
        cfg: &ImageClassifierCfg,
    )->ImageClassifier{
        ImageClassifier{
            backbone: resnet::ResNet::new(p, &cfg.backbone),
            neck: gap::GlobalAveragePooling::new(p, &cfg.neck),
            head: linear_head::LinearClsHead::new(p, &cfg.head),
        }
    }
    pub fn forward_train(
        &self,
        xs: &Tensor,
        ys:&Tensor,
    )-> Tensor
    {
        nn::ModuleT::forward_t(self, xs, true).cross_entropy_for_logits(&ys)
    }
    pub fn simple_test(
        &self,
        xs: &Tensor,
    )-> Tensor
    {
        nn::ModuleT::forward_t(self, xs, false).argmax(-1, false)
    }
}

impl nn::ModuleT for ImageClassifier {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = &self.backbone.forward_t(xs, train);
        let xs = &self.neck.forward_t(xs, train);
        self.head.forward_t(xs, train)
    }
}