use crate::{
    nn,
    Tensor,
};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag="type")]
pub struct GlobalAveragePoolingCfg{}

impl GlobalAveragePoolingCfg {
    pub fn loads(json_str: &String) -> GlobalAveragePoolingCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct GlobalAveragePooling{
}

impl GlobalAveragePooling {
    pub fn new(
        p: &nn::Path,
        cfg: &GlobalAveragePoolingCfg,
    )->GlobalAveragePooling{
        GlobalAveragePooling{}
    }
}

impl nn::ModuleT for GlobalAveragePooling {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.adaptive_avg_pool2d(&[1, 1])
            .flat_view()
    }
}