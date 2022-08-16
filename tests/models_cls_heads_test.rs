use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_cls_heads_linear_head(){
    const IN: i64 = 512;
    const OUT:i64 = 2;
    use river::models::cls_heads::linear_head;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = linear_head::LinearClsHead::new(&vs.root(), IN, OUT);
    let t = Tensor::zeros(&[1,IN], kind::FLOAT_CPU).to_device(vs.device());
    let o_ = net.forward_t(&t, true);
}