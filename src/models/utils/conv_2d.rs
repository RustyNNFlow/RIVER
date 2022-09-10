use crate::{
    nn,
    nn::Conv2D,
};
pub fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride, padding, bias: false, ..Default::default() };
    nn::conv2d(&p, c_in, c_out, ksize, conv2d_cfg)
}