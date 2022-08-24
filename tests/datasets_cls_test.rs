extern crate serde;
extern crate anyhow;
use river::datasets::{
    cls_dataset,
    dataset_iter,
};
#[test]
fn test_classification_dataset_train(){

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
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

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
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

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
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

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        false,
        false,
        None,
        None,
    );
    for (idx, (bimages, blabels, ins_groups)) in obj.iter(1).enumerate(){
        println!("{} {:?} {:?} {:?}", idx, bimages, blabels, ins_groups);
    }

}