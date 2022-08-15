extern crate river;
use river::{
    Device,
    nn,
};
#[test]
fn test_resnet(){
    use river::models::backbones::resnet;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let _net = resnet::ResNet::new(&vs.root(), 2, 2, 2, 2);
}