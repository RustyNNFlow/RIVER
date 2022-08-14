extern crate river;
extern crate serde;
use serde::{Serialize, Deserialize};
#[test]
fn test_instance_group_classification(){
    use river::addons::classification::instance;
    let s = "{\"data\":[{\"category\":\"修勾\",\"score\":0.1,\"type\":\"Infer\"}],\"usage\":{\"image_height\":1,\"image_name\":\"\",\"image_width\":1}}";
    let deserialized:instance::ClsInstancesGroup=instance::ClsInstancesGroup::loads(&String::from(s));
    let _o = deserialized.dumps();
    let _group_name = deserialized.group_name();
    let _instance_num = deserialized.instance_num();

    let s = "{\"data\":[{\"category\":\"修勾\",\"type\":\"Train\"}],\"usage\":{\"image_height\":1,\"image_name\":\"\",\"image_width\":1}}";
    let deserialized:instance::ClsInstancesGroup=instance::ClsInstancesGroup::loads(&String::from(s));
    let _o = deserialized.dumps();
    let _group_name = deserialized.group_name();
    let _instance_num = deserialized.instance_num();
}