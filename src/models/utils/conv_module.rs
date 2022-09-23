use crate::{
    nn,
    nn::SequentialT,
    models::utils::conv_2d::conv2d,
};


pub fn conv_module(
    p: &nn::Path,
    c_in:i64,
    c_out:i64,
    ksize:i64,
    stride:i64,
    padding:i64,
    activation:bool,
) ->SequentialT {
        let mut layer = nn::seq_t()
            .add(conv2d(p / "conv", c_in, c_out, ksize, padding, stride))
            .add(nn::batch_norm2d(p / "bn", c_out, Default::default()));
        if activation{
            layer=layer.add(nn::relu());
        }
        layer
    }