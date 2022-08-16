use crate::{
    models::backbones::resnet,
    models::necks::gap,
    models::cls_heads::linear_head,
    nn,
    Tensor,
};

#[derive(Debug)]
pub struct ImageClassifier{
    backbone: resnet::ResNet,
    neck: gap::GlobalAveragePooling,
    head: linear_head::LinearClsHead,
}

impl ImageClassifier {
    pub fn new(
        p: &nn::Path,
    )->ImageClassifier{
        ImageClassifier{
            backbone: resnet::ResNet::new(p, 2, 2, 2, 2),
            neck: gap::GlobalAveragePooling::new(),
            head: linear_head::LinearClsHead::new(p, 512, 2),
        }
    }
}

impl nn::ModuleT for ImageClassifier {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = &self.backbone.forward_t(xs, train);
        let xs = &self.neck.forward_t(xs, train);
        self.head.forward_t(xs, train)
    }
}