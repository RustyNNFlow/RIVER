use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_backbones_resnet_cls(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 3;
    const B: i64 = 1;
    use river::models::backbones::resnet;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let s = String::from("{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[3],\"num_stages\":4}");
    let cfg:resnet::ResNetCfg = resnet::ResNetCfg::loads(&s);
    let s_ = cfg.dumps();
    let net = resnet::ResNet::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let _o = net.forward_t(&t, true);
}

#[test]
fn test_models_backbones_resnet_det(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 3;
    const B: i64 = 1;
    use river::models::backbones::resnet;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let s = String::from("{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[1,2,3],\"num_stages\":4}");
    let cfg:resnet::ResNetCfg = resnet::ResNetCfg::loads(&s);
    let s_ = cfg.dumps();
    let net = resnet::ResNet::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let _o = net.forward_t(&t, true);
}