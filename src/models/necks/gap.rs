use crate::{
    nn,
    Tensor,
};

#[derive(Debug)]
pub struct GlobalAveragePooling{
}

impl GlobalAveragePooling {
    pub fn new()->GlobalAveragePooling{
        GlobalAveragePooling{}
    }
}

impl nn::ModuleT for GlobalAveragePooling {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.adaptive_avg_pool2d(&[1, 1])
            .flat_view()
    }
}