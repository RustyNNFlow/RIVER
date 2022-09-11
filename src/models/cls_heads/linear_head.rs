use crate::{
    nn,
    nn::Linear,
    Tensor,
};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct LinearClsHeadCfg{
    in_channels:i64,
    num_classes:i64
}

impl LinearClsHeadCfg {
    pub fn loads(json_str: &String) -> LinearClsHeadCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct LinearClsHead{
    fc:Linear,
}

impl LinearClsHead {
    pub fn new(
        p: &nn::Path,
        cfg: &LinearClsHeadCfg,
    )->LinearClsHead{
        LinearClsHead{
            fc: nn::linear(p / "fc", cfg.in_channels, cfg.num_classes, Default::default())
        }
    }
    pub fn forward_t(&self, xs: &Vec<Tensor>, train: bool) -> Tensor {
        xs[xs.len()-1].apply_t(&self.fc, train)
    }
}
