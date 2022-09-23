use river::{
    Device,
    Tensor,
    nn,
    kind,
};
#[test]
fn test_models_losses_focal_loss(){
    let s = String::from("{\"use_sigmoid\":true,\"gamma\":2.0,\"alpha\":0.25,\"loss_weight\":1.0}");
    use river::models::losses::focal_loss;
    let cfg = focal_loss::FocalLossCfg::loads(&s);
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let n = focal_loss::FocalLoss::new(&vs.root(), &cfg);
    let flatten_cls_scores:Tensor = Tensor::zeros(&[256, 80], (kind::Kind::Double, Device::cuda_if_available()));
    let flatten_labels:Tensor = Tensor::ones(&[256, 1], (kind::Kind::Int64, Device::cuda_if_available()))+2;
    let _o = n.forward(flatten_cls_scores, flatten_labels, 256);
}
