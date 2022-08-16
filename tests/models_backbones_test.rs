use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_backbones_resnet(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 3;
    const B: i64 = 1;
    use river::models::backbones::resnet;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let s = String::from("{\"c1\":2,\"c2\":2,\"c3\":2,\"c4\":2,\"in_channels\":3}");
    let cfg:resnet::ResNetCfg = resnet::ResNetCfg::loads(&s);
    let s_ = cfg.dumps();
    let net = resnet::ResNet::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let o_ = net.forward_t(&t, true);
}