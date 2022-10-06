extern crate serde;
extern crate anyhow;
use river::datasets::{
    cls_dataset,
    dataset_iter,
    pipelines::compose::Compose,
};
#[test]
fn test_classification_dataset_train(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        compose,
        true,
        true,
        Some(String::from("tests/assets/classification/dataset/train_anno.json")),
        Some(String::from("tests/assets/classification/dataset/category_info.json")),
    );
    let o = obj.prepare(0);
    let _s = obj.len();
    if let Some(res) = o{
        let _n = res.group_name();
    }

}
#[test]
fn test_classification_dataset_test(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        compose,
        false,
        false,
        None,
        None,
    );
    let o = obj.prepare(0);
    let _s = obj.len();
    if let Some(res) = o{
        let _n = res.group_name();
    }
}

#[test]
fn test_classification_dataset_train_iter(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        compose,
        true,
        true,
        Some(String::from("tests/assets/classification/dataset/train_anno.json")),
        Some(String::from("tests/assets/classification/dataset/category_info.json")),
    );

    for (idx, (bimages, blabels, ins_groups)) in obj.iter(1).enumerate(){
        println!("{} {:?} {:?} {:?}", idx, bimages, blabels, ins_groups);
    }

}

#[test]
fn test_classification_dataset_test_iter(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        compose,
        false,
        false,
        None,
        None,
    );
    for (idx, (bimages, blabels, ins_groups)) in obj.iter(1).enumerate(){
        println!("{} {:?} {:?} {:?}", idx, bimages, blabels, ins_groups);
    }
}

#[test]
fn test_classification_dataset_load_dump(){
    let s = String::from("tests/assets/classification/dataset/cls_dataset_0.json");
    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::load_by_file(&s);
    obj.dump_to_file(
        &std::env::temp_dir()
        .join(format!("det_dataset_0.json"))
        .into_os_string()
        .into_string()
        .unwrap()
    );
}