use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_bbox_head_single(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 512;
    const B: i64 = 1;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"stride\":8}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadSingleCfg = fcos_head::FCOSHeadSingleCfg::loads(&s);
    let n = fcos_head::FCOSHeadSingle::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let o_ = n.forward_t(&t, true);
}

#[test]
fn test_models_bbox_heads(){
    const W: [i64; 3] = [32, 64, 128];
    const H: [i64; 3] = [32, 64, 128];
    const C: i64 = 512;
    const B: i64 = 1;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32]}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);
    let mut ts: Vec<Tensor> = Vec::new();
    for i in 0..3{
        ts.push(Tensor::zeros(&[B,C,H[i],W[i]], kind::FLOAT_CPU).to_device(vs.device()));
    }
    let os_ = n.forward_t(&ts, true);
}

#[test]
fn test_models_bbox_head_get_points_single(){
    const W: i64 = 512;
    const H: i64 = 128;
    const S: i64 = 32;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32]}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);

    let o_ = n.get_points_single(H, W, S);
}