use crate::{
    nn,
    nn::Conv2D,
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
    depth:i64,
    counts: Vec<i64>,
    in_channel:i64,
    stem_channel:i64,
    base_channel:i64,
    out_indices:Vec<i64>,
    num_stages:i64,
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
    stage_blocks: Vec<SequentialT>,
    out_indices:Vec<i64>,
}

impl ResNet {
    pub fn new(
        p: &nn::Path,
       cfg:&ResNetCfg,
    )->ResNet{
        let arch_settings:Vec<i64> = match cfg.depth {
            18=>[2, 2, 2, 2].to_vec(),
            34|50=>[3, 4, 6, 3].to_vec(),
            101=>[3, 4, 23, 3].to_vec(),
            152=>[3, 8, 36, 3].to_vec(),
            _=>[2, 2, 2, 2].to_vec(),
        };
        let strides = [1, 2, 2, 2];
        let mut stage_counts:Vec<i64> = Vec::new();
        for i in 0..cfg.num_stages{
            stage_counts.push(
                arch_settings[i as usize]
            );
        }
        let mut stage_blocks:Vec<SequentialT>=Vec::new();
        let mut inplanes=cfg.base_channel;
        let mut planes;
        for (i,c) in stage_counts.iter().enumerate(){
            planes = cfg.base_channel * 2_i64.pow(i as u32);
            stage_blocks.push(
                basic_layer(
                    p / format!("layer{}", i+1),
                    inplanes,
                    planes,
                    strides[i],
                    *c),
            );
            inplanes = planes;
        }

        ResNet{
            conv1: conv2d(p / "conv1", cfg.in_channel, cfg.stem_channel, 7, 3, 2),
            bn1: nn::batch_norm2d(p / "bn1", 64, Default::default()),
            stage_blocks:stage_blocks,
            out_indices:cfg.out_indices.clone(),
        }
    }
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Vec<Tensor> {
        let mut x = xs.apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false);
        let mut outs:Vec<Tensor>=Vec::new();
        for i in 0..self.stage_blocks.len(){
            x = x.apply_t(&self.stage_blocks[i], train);
            if self.out_indices.iter().any(|&index| index==(i as i64)) {
                outs.push(
                    x.copy()
                );
            }
        }
        outs
    }
}
