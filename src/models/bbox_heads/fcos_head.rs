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

fn conv_module(
    p: &nn::Path,
    c_in:i64,
    c_out:i64,
    ksize:i64,
    stride:i64,
    padding:i64,
)
    ->SequentialT {
    nn::seq_t()
        .add(conv2d(p / "conv", c_in, c_out, ksize, padding, stride))
        .add(nn::batch_norm2d(p / "bn", c_out, Default::default()))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct FCOSHeadSingleCfg{
    in_channels:i64,
    num_classes:i64,
    feat_channels:i64,
    stacked_convs:i64,
    stride:i64,
}

impl FCOSHeadSingleCfg {
    pub fn loads(json_str: &String) -> FCOSHeadSingleCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn new(
        in_channels:i64,
        num_classes:i64,
        feat_channels:i64,
        stacked_convs:i64,
        stride:i64,
    )->FCOSHeadSingleCfg{
        FCOSHeadSingleCfg{
            in_channels:in_channels,
            num_classes:num_classes,
            feat_channels:feat_channels,
            stacked_convs:stacked_convs,
            stride:stride,
        }
    }
}

#[derive(Debug)]
pub struct FCOSHeadSingle{
    hybrid_convs:SequentialT,
    fcos_cls:Conv2D,
    fcos_reg:Conv2D,
}

impl FCOSHeadSingle {
    pub fn new(
        p: &nn::Path,
        cfg: &FCOSHeadSingleCfg,
    )->FCOSHeadSingle{
        let cls_out_channels = cfg.num_classes - 1;

        let mut hybrid_convs = nn::seq_t();
        for i in 0..cfg.stacked_convs{
            let c_in = match i {
              0=>  cfg.in_channels,
                _=>cfg.feat_channels,
            };
            hybrid_convs=hybrid_convs.add(
                conv_module(
                p,
                c_in,
                cfg.feat_channels,
                3,
                1,
                1,
                )
            );
        }


        let fcos_cls:Conv2D= conv2d(
                p / "fcos_cls",
                cfg.feat_channels,
                cls_out_channels,
                3,
                1,
                1,
        );

        let fcos_reg =conv2d(
                p / "fcos_reg",
                cfg.feat_channels,
                4,
                3,
                1,
                1,
        );

        FCOSHeadSingle{
            hybrid_convs: hybrid_convs,
            fcos_cls: fcos_cls,
            fcos_reg: fcos_reg,
        }
    }
}

impl nn::ModuleT for FCOSHeadSingle {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let feat=xs.apply_t(&self.hybrid_convs, train);
        let mut outs:Vec<Tensor> = Vec::new();
        outs.push(feat.apply_t(&self.fcos_cls, train));
        outs.push(feat.apply_t(&self.fcos_reg, train));
        Tensor::cat(&outs, 1)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct FCOSHeadCfg{
    in_channels:i64,
    num_classes:i64,
    feat_channels:i64,
    stacked_convs:i64,
    strides:Vec<i64>,
}

impl FCOSHeadCfg {
    pub fn loads(json_str: &String) -> FCOSHeadCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct FCOSHead{
    heads:Vec<FCOSHeadSingle>,
}

impl FCOSHead {
    pub fn new(
        p: &nn::Path,
        cfg: &FCOSHeadCfg,
    )->FCOSHead{
        let cls_out_channels = cfg.num_classes - 1;
        let mut heads:Vec<FCOSHeadSingle>=Vec::new();

        for s in cfg.strides.iter() {
            let cfg_s:FCOSHeadSingleCfg = FCOSHeadSingleCfg::new(
                cfg.in_channels,
                cfg.num_classes,
                cfg.feat_channels,
                cfg.stacked_convs,
                *s,
            );
            heads.push(FCOSHeadSingle::new(p, &cfg_s));
        }
        FCOSHead{
            heads:heads,
        }
    }
}

// impl nn::ModuleT for FCOSHead {
//     fn forward_t(&self, xs: &Tensor, train: bool) -> Vec<Tensor> {
//         xs
//     }
// }