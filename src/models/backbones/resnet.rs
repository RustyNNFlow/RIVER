use crate::{
    nn,
    nn::Conv2D,
    nn::FuncT,
    nn::ModuleT,
    nn::BatchNorm,
    nn::SequentialT,
    Tensor,
};
use serde::{Serialize, Deserialize};

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride, padding, bias: false, ..Default::default() };
    nn::conv2d(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    if stride != 1 || c_in != c_out {
        nn::seq_t().add(conv2d(&p / "0", c_in, c_out, 1, 0, stride)).add(nn::batch_norm2d(
            &p / "1",
            c_out,
            Default::default(),
        ))
    } else {
        nn::seq_t()
    }
}

fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    let conv1 = conv2d(&p / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = nn::batch_norm2d(&p / "bn1", c_out, Default::default());
    let conv2 = conv2d(&p / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = nn::batch_norm2d(&p / "bn2", c_out, Default::default());
    let downsample = downsample(&p / "downsample", c_in, c_out, stride);
    nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv1).apply_t(&bn1, train).relu().apply(&conv2).apply_t(&bn2, train);
        (xs.apply_t(&downsample, train) + ys).relu()
    })
}

fn basic_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> SequentialT {
    let mut layer = nn::seq_t().add(basic_block(&p / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&p / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct ResNetCfg{
    c1: i64,
    c2: i64,
    c3: i64,
    c4: i64,
    in_channels:i64,
}

impl ResNetCfg {
    pub fn loads(json_str: &String) -> ResNetCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct ResNet{
    conv1: Conv2D,
    bn1: BatchNorm,
    layer1: SequentialT,
    layer2: SequentialT,
    layer3: SequentialT,
    layer4: SequentialT,
}

impl ResNet {
    pub fn new(
        p: &nn::Path,
       cfg:&ResNetCfg,
    )->ResNet{
        ResNet{
            conv1: conv2d(p / "conv1", cfg.in_channels, 64, 7, 3, 2),
            bn1: nn::batch_norm2d(p / "bn1", 64, Default::default()),
            layer1: basic_layer(p / "layer1", 64, 64, 1, cfg.c1),
            layer2:  basic_layer(p / "layer2", 64, 128, 2, cfg.c2),
            layer3: basic_layer(p / "layer3", 128, 256, 2, cfg.c3),
            layer4: basic_layer(p / "layer4", 256, 512, 2, cfg.c4),
        }
    }
}

impl nn::ModuleT for ResNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
            .apply_t(&self.layer1, train)
            .apply_t(&self.layer2, train)
            .apply_t(&self.layer3, train)
            .apply_t(&self.layer4, train)
    }
}