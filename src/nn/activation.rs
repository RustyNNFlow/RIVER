//! A linear fully-connected layer.
use crate::Tensor;
use std::borrow::Borrow;


/// A linear fully-connected layer.
#[derive(Debug)]
pub struct Relu {}

pub fn relu<'a>(
) -> Relu {
    Relu{}
}

impl super::module::Module for Relu {
    fn forward(&self, xs: &Tensor) -> Tensor {
            xs.relu()
    }
}
