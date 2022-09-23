use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_cls_heads_linear_head(){
    use river::models::cls_heads::linear_head;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let s = String::from("{\"in_channels\":512,\"num_classes\":2}");
    let cfg:linear_head::LinearClsHeadCfg = linear_head::LinearClsHeadCfg::loads(&s);
    let s_ = cfg.dumps();

    let net = linear_head::LinearClsHead::new(&vs.root(), &cfg);
    let mut ts:Vec<Tensor>=Vec::new();
    ts.push(Tensor::zeros(&[1,512], kind::FLOAT_CPU).to_device(vs.device()));
    let _o = net.forward_t(&ts, true);
}