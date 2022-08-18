use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag="type")]
pub enum ClsInstanceData{
    Train {category:String},
    Infer {category:String, score:f64},
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClsGroupUsage{
    image_name:String,
    image_height:i32,
    image_width:i32,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ClsInstancesGroup{
    data:Vec<ClsInstanceData>,
    usage:ClsGroupUsage,
}

impl ClsInstancesGroup{
    pub fn loads(json_str: &String)->ClsInstancesGroup{
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
    pub fn image_height(&self)->i32{
        self.usage.image_height
    }
    pub fn image_width(&self)->i32{
        self.usage.image_width
    }
    pub fn new(
        category:&String,
        image_name:&String,
        image_height:i32,
        image_width:i32,
        score:f64,
    )->ClsInstancesGroup
    {
        let mut vec_data: Vec<ClsInstanceData>= Vec::new();
        vec_data.push(
            ClsInstanceData::Infer
            {
                category:category.clone(),
                score:score,
            }
        );
        ClsInstancesGroup{
            data:vec_data,
            usage:ClsGroupUsage{
                image_name:image_name.clone(),
                image_height:image_height,
                image_width:image_width,
            },
        }
    }
}