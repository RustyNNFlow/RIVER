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

#[test]
fn test_instance_dataset_classification(){
    use river::addons::classification::instance_dataset;
    let s = "{\"data\":[{\"data\":[{\"category\":\"修勾\",\"score\":0.1,\"type\":\"Infer\"}],\"usage\":{\"image_height\":1,\"image_name\":\"\",\"image_width\":1}}],\"usage\":{\"dataset_name\":\"1\"}}";
    let deserialized:instance_dataset::ClsInstancesDataset=instance_dataset::ClsInstancesDataset::loads(&String::from(s));
    let _o = deserialized.dumps();
    let _dataset_name = deserialized.dataset_name();

}