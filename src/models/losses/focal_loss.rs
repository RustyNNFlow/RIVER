use crate::{
    nn,
    Tensor,
    Reduction,
    kind,
};
use serde::{Serialize, Deserialize};

pub fn sigmoid_focal_loss(
    pred:Tensor,
    target:Tensor,
    gamma:f64,
    alpha:f64,
    avg_factor:i64,
)->Tensor{
    let device = pred.device();
    let pred_sigmoid = pred.sigmoid();

    let target = target.type_as(&pred);
    let pt:Tensor = Tensor::of_slice(&[1.])
        .to_device(device)
        .g_add(&pred_sigmoid.neg())
        .g_mul(&target)
        .g_add(&
            Tensor::of_slice(&[1.])
                .to_device(device)
                .g_add(& target.neg())
                .g_mul(&pred_sigmoid)
        );

    let focal_weight = (Tensor::of_slice(&[alpha])
        .to_device(device)
        .g_mul(&target)
        .g_add(&
            Tensor::of_slice(&[1. - alpha]).to_device(device)
                .g_mul(&Tensor::of_slice(&[1.]).to_device(device).g_sub(& target))
                )).g_mul(&pt.pow(&Tensor::of_slice(&[gamma]).to_device(device)));
    let loss:Tensor = pred.binary_cross_entropy_with_logits::<Tensor>(
        &target,
        None,
        None,
        Reduction::Mean,
    ).g_mul(&focal_weight);
    loss.sum(kind::Kind::Double).true_divide_scalar(avg_factor)
}
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct FocalLossCfg{
    use_sigmoid:bool,
    gamma:f64,
    alpha:f64,
    loss_weight:f64,
}

impl FocalLossCfg {
    pub fn loads(json_str: &String) -> FocalLossCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct FocalLoss{
    gamma:f64,
    alpha:f64,
    loss_weight:f64,
}

impl FocalLoss {
    pub fn new(
        cfg: &FocalLossCfg,
    )->FocalLoss{
        FocalLoss{
            gamma:cfg.gamma,
            alpha:cfg.alpha,
            loss_weight:cfg.loss_weight,
        }
    }
    pub fn forward(
        &self,
        pred:Tensor,
        target:Tensor,
        avg_factor:i64,
    ) -> Tensor {
        let num_classes = pred.size()[1];
        let mut target = target.one_hot(num_classes+1).squeeze_dim(1);
        target = target.narrow(1, 1, num_classes);
        self.loss_weight*sigmoid_focal_loss(pred, target, self.gamma, self.alpha, avg_factor)
    }
}



