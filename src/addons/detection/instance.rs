use serde::{Serialize, Deserialize};
use crate::datasets::category_info;
use crate::tensor::Tensor;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BBox{
    pub x1:f64,
    pub y1:f64,
    pub x2:f64,
    pub y2:f64,
}
impl BBox{
    pub fn to_vec(&self)->Vec<f64>{
        let mut vec:Vec<f64> = Vec::new();
        vec.push(self.x1);
        vec.push(self.y1);
        vec.push(self.x2);
        vec.push(self.y2);
        vec
    }
    pub fn to_tensor(&self)->Tensor{
        let mut vec:Vec<f64> = Vec::new();
        vec.push(self.x1);
        vec.push(self.y1);
        vec.push(self.x2);
        vec.push(self.y2);
        Tensor::of_slice(&vec)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct  DetInstanceData{
    category:String,
    pub bbox:BBox,
    score:Option<f64>,
    pub category_index:Option<i64>,
}
impl DetInstanceData{
    pub fn update_category_index(&mut self, index:i64){
        self.category_index = Some(index);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetGroupUsage{
    image_name:String,
    image_height:i64,
    image_width:i64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetInstancesGroup{
    pub data:Vec<DetInstanceData>,
    usage:DetGroupUsage,
}

impl DetInstancesGroup{
    pub fn loads(json_str: &String)->DetInstancesGroup{
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self)->String{
        serde_json::to_string(&self).unwrap()
    }
    pub fn group_name(&self)->String{
        self.usage.image_name.clone()
    }
    pub fn instance_num(&self)->usize{
        self.data.len()
    }
    pub fn image_height(&self)->i64{
        self.usage.image_height
    }
    pub fn image_width(&self)->i64{
        self.usage.image_width
    }
    pub fn update_category_index(&mut self, category_info: category_info::CategoryInfo){
        for d in self.data.iter_mut(){
            let idx=category_info.cat2id.get(&d.category).unwrap();
            d.update_category_index(*idx);
        }

    }
    pub fn update_image_height(&mut self, image_height:i64){
        self.usage.image_height=image_height;
    }
    pub fn update_image_width(&mut self, image_width:i64){
        self.usage.image_width=image_width;
    }
    pub fn resize_bbox(&mut self, fx:f64, fy:f64){
        let len = self.data.len();
        for i in 0..len{
            self.data[i].bbox.x1 = self.data[i].bbox.x1*fx;
            self.data[i].bbox.y1 = self.data[i].bbox.y1*fy;
            self.data[i].bbox.x2 = self.data[i].bbox.x2*fx;
            self.data[i].bbox.y2 = self.data[i].bbox.y2*fy;
        }
    }
    pub fn new(
        vec_data:Vec<DetInstanceData>,
        image_name:&String,
        image_height:i64,
        image_width:i64,
        score:f64,
    )->DetInstancesGroup
    {
        DetInstancesGroup{
            data:vec_data,
            usage:DetGroupUsage{
                image_name:image_name.clone(),
                image_height:image_height,
                image_width:image_width,
            },
        }
    }
}