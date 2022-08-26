use crate::{
    Tensor,
    addons::classification::instance::ClsInstancesGroup,
    addons::classification::instance_dataset::ClsInstancesDataset,
    vision::image as image_op,
    Kind,
    datasets::category_info,
    datasets::dataset_iter,
};
use std::path::Path;

#[derive(Debug)]
pub struct DatasetResult{
    pub img:Tensor,
    pub instances_group:ClsInstancesGroup,
}
impl DatasetResult{
    pub fn group_name(&self)->String{
        self.instances_group.group_name()
    }
    pub fn image_height(&self)->i64{
        self.instances_group.image_height()
    }
    pub fn image_width(&self)->i64{
        self.instances_group.image_width()
    }
    pub fn x(&self)->Tensor{
        self.img.shallow_clone()
    }
    pub fn y(&self)->Tensor{
        let mut ys:Vec<i64> = Vec::new();
        for d in self.instances_group.data.iter(){
            if let Some(s) = d.category_index{
                ys.push(s);
            }
        }
        Tensor::of_slice(&ys)
    }
    pub fn update_img(&mut self, img:Tensor){
        self.img = img;
    }
    pub fn update_image_height(&mut self, image_height:i64){
        self.instances_group.update_image_height(image_height);
    }
    pub fn update_image_width(&mut self, image_width:i64){
        self.instances_group.update_image_width(image_width);
    }
    pub fn new(
        img_root:&String,
        img_name:&String,
    )->DatasetResult{
        let path = Path::new(&img_root).join(&img_name);
        let img = image_op::load(path).unwrap();
        let size = img.size();
        let height = size[1];
        let width = size[2];
        let ins_group:ClsInstancesGroup = ClsInstancesGroup::new(
            &String::from(""),
            img_name,
            height,
            width,
            0.0,
        );
        DatasetResult{
            img:img,
            instances_group:ins_group,
        }
    }
}