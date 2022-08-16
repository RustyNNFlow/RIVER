use crate::{
    nn,
    nn::Linear,
    Tensor,
};

#[derive(Debug)]
pub struct LinearClsHead{
    fc:Linear,
}

impl LinearClsHead {
    pub fn new(
        p: &nn::Path,
        in_channels:i64,
        num_classes:i64
    )->LinearClsHead{
        LinearClsHead{
            fc: nn::linear(p / "fc", in_channels, num_classes, Default::default())
        }
    }
}

impl nn::ModuleT for LinearClsHead {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply_t(&self.fc, train)
    }
}