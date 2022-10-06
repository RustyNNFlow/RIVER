use crate::{
    Tensor,
    addons::detection::instance::DetInstancesGroup,
    addons::detection::instance::DetInstanceData,
    addons::detection::instance::BBox,
    vision::image as image_op,
};
use std::path::Path;

#[derive(Debug)]
pub struct DetDatasetResult{
    pub img:Tensor,
    pub instances_group:DetInstancesGroup,
}
impl DetDatasetResult{
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
    pub fn gt_labels(&self)->Tensor{
        let mut ys:Vec<i64> = Vec::new();
        for d in self.instances_group.data.iter(){
            if let Some(s) = d.category_index{
                ys.push(s);
            }
        }
        Tensor::of_slice(&ys)
    }
    pub fn gt_bboxes(&self)->Tensor{
        let mut ys:Vec<Tensor> = Vec::new();
        for d in self.instances_group.data.iter(){
            if let s = &d.bbox{
                ys.push(s.to_tensor());
            }
        }
        if ys.len() > 0 {
            return Tensor::cat(&ys, 0);
        }
        else {
            let empty:Vec<f64>=Vec::new();
            return Tensor::of_slice(&empty);
        }

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
    pub fn resize_bbox(&mut self, fx:f64, fy:f64){
        self.instances_group.resize_bbox(fx, fy);
    }

    pub fn new(
        img_root:&String,
        img_name:&String,
    )->DetDatasetResult{
        let path = Path::new(&img_root).join(&img_name);
        let img = image_op::load(path).unwrap();
        let size = img.size();
        let height = size[1];
        let width = size[2];
        let vec_data:Vec<DetInstanceData> = Vec::new();
        let ins_group:DetInstancesGroup = DetInstancesGroup::new(
            vec_data,
            img_name,
            height,
            width,
            0.0,
        );


        DetDatasetResult{
            img:img,
            instances_group:ins_group,
        }
    }
}