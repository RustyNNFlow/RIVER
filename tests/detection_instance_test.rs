extern crate river;
extern crate serde;
use serde::{Serialize, Deserialize};
#[test]
fn test_instance_group_detection(){
    use river::addons::detection::instance;
    let s = "{\"data\":[{\"category\":\"修勾\",\"bbox\":{\"x1\":1.0,\"y1\":2.0,\"x2\":3.0,\"y2\":4.0},\"score\":0.1}],\"usage\":{\"image_height\":10,\"image_name\":\"\",\"image_width\":10}}";
    let deserialized:instance::DetInstancesGroup=instance::DetInstancesGroup::loads(&String::from(s));
    let _o = deserialized.dumps();
    let _group_name = deserialized.group_name();
    let _instance_num = deserialized.instance_num();

    let s = "{\"data\":[{\"category\":\"修勾\",\"bbox\":{\"x1\":1.0,\"y1\":2.0,\"x2\":3.0,\"y2\":4.0}}],\"usage\":{\"image_height\":10,\"image_name\":\"\",\"image_width\":10}}";
    let deserialized:instance::DetInstancesGroup=instance::DetInstancesGroup::loads(&String::from(s));
    let _o = deserialized.dumps();
    let _group_name = deserialized.group_name();
    let _instance_num = deserialized.instance_num();

}

#[test]
fn test_instance_dataset_detection(){
    use river::addons::detection::instance_dataset;

    let deserialized:instance_dataset::DetInstancesDataset=
        instance_dataset::DetInstancesDataset::load_by_file(&String::from("tests/assets/detection/instance_dataset_0.json"));
    let _o = deserialized.dumps();
    let _dataset_name = deserialized.dataset_name();
    println!("{:?}", deserialized);


    let deserialized:instance_dataset::DetInstancesDataset=
        instance_dataset::DetInstancesDataset::load_by_file(&String::from("tests/assets/detection/instance_dataset_1.json"));
    let _o = deserialized.dumps();
    let _dataset_name = deserialized.dataset_name();

    deserialized.dump_to_file(&std::env::temp_dir().join(format!("tmp.json")).into_os_string().into_string().unwrap());

}
