//! A linear fully-connected layer.
use crate::Tensor;
use std::borrow::Borrow;

/// Configuration for a scale layer.
#[derive(Debug, Clone, Copy)]
pub struct ScaleConfig {
    pub scale_init: super::Init,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        ScaleConfig { scale_init: super::Init::Const(1.)}
    }
}

/// A linear fully-connected layer.
#[derive(Debug)]
pub struct Scale {
    pub s: Tensor,
}

/// Creates a new linear layer.
pub fn scale<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    c: ScaleConfig,
) -> Scale {
    let vs = vs.borrow();
    Scale { s: vs.var("scale", &[1], c.scale_init)}
}

impl super::module::Module for Scale {
    fn forward(&self, xs: &Tensor) -> Tensor {
            xs.multiply(&self.s)
    }
}
