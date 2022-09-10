use crate::{
    nn,
    nn::Conv2D,
    nn::FuncT,
    nn::ModuleT,
    nn::BatchNorm,
    nn::SequentialT,
    nn::Init,
    nn::Scale,
    Tensor,
    models::utils::conv_module::conv_module,
    Kind,
};
use serde::{Serialize, Deserialize};
use std::borrow::Borrow;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct BiFPNModuleCfg{
    channels:i64,
    levels:i64,
}

impl BiFPNModuleCfg {
    pub fn loads(json_str: &String) -> BiFPNModuleCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct BiFPNModule{
    levels:i64,
    bifpn_convs:Vec<SequentialT>,
    w1:Tensor,
    w2:Tensor,
    eps:f64,
}

impl BiFPNModule {
    pub fn new(
        p: &nn::Path,
        cfg: &BiFPNModuleCfg,
    )->BiFPNModule{
        let mut bifpn_convs:Vec<SequentialT> = Vec::new();
        for _ in 0..2{
            for _ in 0..cfg.levels{
                bifpn_convs.push(
                    nn::seq_t().add(
                        conv_module(
                            p,
                            cfg.channels,
                            cfg.channels,
                            3,
                            1,
                            1,
                            true,
                        )
                    )
                );
            };
        }
        let vs = p.borrow();
        let init = Init::Const(0.5);
        let w1 = vs.var("w1", &[2, cfg.levels], init);
        let w2 = vs.var("w1", &[3, cfg.levels-2], init);
        BiFPNModule{
            levels:cfg.levels,
            bifpn_convs:bifpn_convs,
            w1:w1,
            w2:w2,
            eps:0.0001,
        }
    }
    pub fn forward(&self, mut xs: Vec<Tensor>, train: bool) -> Vec<Tensor> {
        assert_eq!(self.levels as usize, xs.len());
        let w1 = &self.w1.relu();
        let w1 = w1/w1.sum_dim_intlist(&[0], false, Kind::Float)+self.eps;
        let w2 = &self.w2.relu();
        let w2 = w2/w2.sum_dim_intlist(&[0], false, Kind::Float)+self.eps;
        let mut idx_bifpn = 0;

        for i in (1..self.levels).rev(){

            let size = xs[i as usize].size();
            let tmp_w=size[3]*2;
            let tmp_h=size[2]*2;

            xs[(i-1) as usize] = (
                self.w1.narrow(0, 0, 1).squeeze_dim(0).narrow(0, i - 1, 1).squeeze_dim(0) * &xs[(i - 1) as usize] +
                    self.w1.narrow(0,1,1).squeeze_dim(0).narrow(0, i - 1, 1).squeeze_dim(0)*&xs[i as usize].
                        upsample_nearest2d(&[tmp_h, tmp_w], 2.0, 2.0)
            )/(
                self.w1.narrow(0,0,1).squeeze_dim(0).narrow(0,i - 1, 1).squeeze_dim(0) +
                    self.w1.narrow(0,1,1).squeeze_dim(0).narrow(0,i - 1, 1).squeeze_dim(0)
                    +self.eps
            );
            xs[(i-1) as usize]=xs[(i-1) as usize].apply_t(&self.bifpn_convs[idx_bifpn], train);
            idx_bifpn = idx_bifpn + 1
        }
        for i in 0..self.levels-2{
            xs[(i+1) as usize]= (&xs[(i+1) as usize]+
                self.w2.narrow(0,1,1).squeeze_dim(0).narrow(0,i,1).squeeze_dim(0)*&xs[i as usize]
                    .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))/
                (
                    self.w2.narrow(0,0,1).squeeze_dim(0).narrow(0,i,1).squeeze_dim(0)
                    +self.w2.narrow(0,1,1).squeeze_dim(0).narrow(0,i,1).squeeze_dim(0)
                    +self.w2.narrow(0,2,1).squeeze_dim(0).narrow(0,i,1).squeeze_dim(0)
                    +self.eps
                );
            xs[(i+1) as usize] = xs[(i+1) as usize].apply_t(&self.bifpn_convs[idx_bifpn], train);
            idx_bifpn = idx_bifpn + 1;

        }
        xs[(self.levels-1) as usize]=(
            self.w1.narrow(0,0,1).squeeze_dim(0).narrow(0,self.levels-1,1).squeeze_dim(0)
                *&xs[(self.levels-1) as usize]
                + self.w1.narrow(0,1,1).squeeze_dim(0).narrow(0,self.levels-1,1).squeeze_dim(0)
                *&xs[(self.levels-2) as usize].
                max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
            )/(
            self.w1.narrow(0,0,1).squeeze_dim(0).narrow(0, self.levels-1, 1).squeeze_dim(0)+
                self.w1.narrow(0,1,1).squeeze_dim(0).narrow(0, self.levels-1, 1).squeeze_dim(0)+self.eps
            );
        xs[(self.levels-1) as usize]=xs[(self.levels-1) as usize].
            apply_t(&self.bifpn_convs[idx_bifpn], train);
        xs
    }
}



#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct BiFPNCfg{
    in_channels:Vec<i64>,
    out_channel:i64,
    num_outs:i64,
    start_level:i64,
    end_level:i64,
    stack:i64,
    add_extra_convs:bool,
    extra_convs_on_inputs:bool,
    relu_before_extra_convs:bool,
    no_norm_on_lateral:bool,
}

impl BiFPNCfg {
    pub fn loads(json_str: &String) -> BiFPNCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}
#[derive(Debug)]
pub struct BiFPN{
    lateral_convs:Vec<SequentialT>,
    stack_bifpn_convs:Vec<BiFPNModule>,
    fpn_convs:SequentialT,
}

impl BiFPN {
    pub fn new(
        p: &nn::Path,
        cfg: &BiFPNCfg,
    )->BiFPN{
        let num_ins=cfg.in_channels.len() as i64;
        let backbone_end_level= match cfg.end_level {
            -1=> num_ins - cfg.start_level,
            _=>cfg.end_level,
        };

        let mut lateral_convs: Vec<SequentialT> = Vec::new();
        for i in cfg.start_level..backbone_end_level{
            lateral_convs.push(
                conv_module(
                    p,
                    cfg.in_channels[i as usize],
                    cfg.out_channel,
                    1,
                    1,
                    0,
                    true,
                )
            );
        }
        let mut stack_bifpn_convs:Vec<BiFPNModule> = Vec::new();
        for _ in 0..cfg.stack{
            let bifpn_module_cfg = BiFPNModuleCfg{
                channels:cfg.out_channel,
                levels:backbone_end_level - cfg.start_level,
            };
            stack_bifpn_convs.push(
                BiFPNModule::new(
                    p,
                    &bifpn_module_cfg,
                )
            );
        }
        let extra_levels=cfg.num_outs - backbone_end_level + cfg.start_level;
        let mut fpn_convs=nn::seq_t();
        if cfg.add_extra_convs&&extra_levels>=1{
            for i in 0..extra_levels {
                let mut in_channel=0;
                if i == 0 && cfg.extra_convs_on_inputs{
                    in_channel = cfg.in_channels[(backbone_end_level-1) as usize];
                }
                else {
                    in_channel = cfg.out_channel
                }
                fpn_convs=fpn_convs.add(
                    conv_module(
                        p,
                        in_channel,
                        cfg.out_channel,
                        3,
                        2,
                        1,
                        true,
                    )
                );
            }
        }


        BiFPN{
            lateral_convs:lateral_convs,
            stack_bifpn_convs:stack_bifpn_convs,
            fpn_convs:fpn_convs,
        }
    }
}