use crate::{
    nn,
    Tensor,
};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
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
    pub fn forward_t(&self, xs: &Vec<Tensor>, train: bool) -> Vec<Tensor> {
        let mut outs:Vec<Tensor> = Vec::new();
        outs.push(
            xs[xs.len()-1].adaptive_avg_pool2d(&[1, 1])
            .flat_view()
        );
        outs
    }
}



