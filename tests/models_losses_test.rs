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
    // let vs = nn::VarStore::new(Device::cuda_if_available());
    let n = focal_loss::FocalLoss::new(&cfg);
    let flatten_cls_scores:Tensor = Tensor::zeros(&[256, 80], (kind::Kind::Double, Device::cuda_if_available()));
    let flatten_labels:Tensor = Tensor::ones(&[256, 1], (kind::Kind::Int64, Device::cuda_if_available()))+2;
    let _o = n.forward(flatten_cls_scores, flatten_labels, 256);
}

#[test]
fn test_models_losses_iou_loss(){
    let s = String::from("{\"eps\":0.001,\"loss_weight\":1.0}");
    use river::models::losses::iou_loss;
    let cfg = iou_loss::IoULossCfg::loads(&s);
    let n = iou_loss::IoULoss::new(&cfg);
    // let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    // let t1 = Tensor::of_slice(&vec);

    let pred:Tensor = Tensor::of_slice(&[1, 3, 21, 20, 1, 3, 21, 20].to_vec()).reshape(&[2,4]);
    let target:Tensor = Tensor::of_slice(&[2, 4, 31, 30, 2, 4, 31, 30].to_vec()).reshape(&[2,4]);
    let avg=pred.size()[0];
    let _o = n.forward(pred, target, avg);
}