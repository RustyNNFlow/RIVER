use crate::{
    nn,
    nn::Conv2D,
    nn::SequentialT,
    nn::Scale,
    Tensor,
    models::utils::conv_module::conv_module,
    models::utils::conv_2d::conv2d,
    kind,
};
use serde::{Serialize, Deserialize};



#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct FCOSHeadSingleCfg{
    in_channels:i64,
    num_classes:i64,
    feat_channels:i64,
    stacked_convs:i64,
    stride:i64,
}

impl FCOSHeadSingleCfg {
    pub fn loads(json_str: &String) -> FCOSHeadSingleCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn new(
        in_channels:i64,
        num_classes:i64,
        feat_channels:i64,
        stacked_convs:i64,
        stride:i64,
    )->FCOSHeadSingleCfg{
        FCOSHeadSingleCfg{
            in_channels:in_channels,
            num_classes:num_classes,
            feat_channels:feat_channels,
            stacked_convs:stacked_convs,
            stride:stride,
        }
    }
}

#[derive(Debug)]
pub struct FCOSHeadSingle{
    hybrid_convs:SequentialT,
    fcos_cls:Conv2D,
    fcos_reg:Conv2D,
    scale:Scale,
}

impl FCOSHeadSingle {
    pub fn new(
        p: &nn::Path,
        cfg: &FCOSHeadSingleCfg,
    )->FCOSHeadSingle{
        let cls_out_channels = cfg.num_classes - 1;

        let mut hybrid_convs = nn::seq_t();
        for i in 0..cfg.stacked_convs{
            let c_in = match i {
              0=>  cfg.in_channels,
                _=>cfg.feat_channels,
            };
            hybrid_convs=hybrid_convs.add(
                conv_module(
                p,
                c_in,
                cfg.feat_channels,
                3,
                1,
                1,
                true,
                )
            );
        }


        let fcos_cls:Conv2D= conv2d(
                p / "fcos_cls",
                cfg.feat_channels,
                cls_out_channels,
                3,
                1,
                1,
        );

        let fcos_reg =conv2d(
                p / "fcos_reg",
                cfg.feat_channels,
                4,
                3,
                1,
                1,
        );

        let scale_cfg = nn::ScaleConfig::default();
        let scale = nn::scale(p, scale_cfg);
        FCOSHeadSingle{
            hybrid_convs: hybrid_convs,
            fcos_cls: fcos_cls,
            fcos_reg: fcos_reg,
            scale:scale,
        }
    }
}

impl nn::ModuleT for FCOSHeadSingle {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let feat=xs.apply_t(&self.hybrid_convs, train);
        let mut outs:Vec<Tensor> = Vec::new();
        outs.push(feat.apply_t(&self.fcos_cls, train));
        outs.push(feat.apply_t(&self.fcos_reg, train).apply_t(&self.scale, train).exp());
        Tensor::cat(&outs, 1)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct FCOSHeadCfg{
    in_channels:i64,
    num_classes:i64,
    feat_channels:i64,
    stacked_convs:i64,
    strides:Vec<i64>,
}

impl FCOSHeadCfg {
    pub fn loads(json_str: &String) -> FCOSHeadCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct FCOSHead{
    heads:Vec<FCOSHeadSingle>,
}

impl FCOSHead {
    pub fn new(
        p: &nn::Path,
        cfg: &FCOSHeadCfg,
    )->FCOSHead{
        let cls_out_channels = cfg.num_classes - 1;
        let mut heads:Vec<FCOSHeadSingle>=Vec::new();

        for s in cfg.strides.iter() {
            let cfg_s:FCOSHeadSingleCfg = FCOSHeadSingleCfg::new(
                cfg.in_channels,
                cfg.num_classes,
                cfg.feat_channels,
                cfg.stacked_convs,
                *s,
            );
            heads.push(FCOSHeadSingle::new(p, &cfg_s));
        }
        FCOSHead{
            heads:heads,
        }
    }
    pub fn forward_t(
            &self,
            xs:&Vec<Tensor>,
            train: bool,
        )->Vec<Tensor>{
        let n = self.heads.len();
        assert_eq!(n, xs.len());
        let mut outs: Vec<Tensor>=Vec::new();
        for i in 0..n{
            let x = &xs[i];
            let y = x.apply_t(&self.heads[i],  train);
            outs.push(y);
        }
        outs
    }
    pub fn get_points_single(
        &self,
        height:i64,
        width:i64,
        stride:i64,
    )->Tensor{
        let mut vec_tensor: Vec<Tensor> = Vec::new();
        vec_tensor.push(Tensor::arange_start_step(0, height*stride, stride, kind::FLOAT_CPU));
        vec_tensor.push(Tensor::arange_start_step(0, width*stride, stride, kind::FLOAT_CPU));
        let mut vec_y_x:Vec<Tensor> = Tensor::meshgrid(&vec_tensor);
        let mut vec_y_x_reshape: Vec<Tensor> = Vec::new();
        for m in vec_y_x.iter(){
            vec_y_x_reshape.push(m.reshape(&[-1]));
        }
        let points = Tensor::stack(&vec_y_x_reshape, -1)+ stride / 2;
        points

    }
    pub fn get_points(
        &self,
        vec_height:Vec<i64>,
        vec_width:Vec<i64>,
        vec_stride:Vec<i64>,
    )->Vec<Tensor>{
        let n:usize = vec_stride.len();
        assert_eq!(n, vec_height.len());
        assert_eq!(n, vec_width.len());
        let mut vec_points:Vec<Tensor> = Vec::new();
        for i in 0..n{
            vec_points.push(
                self.get_points_single(
                    vec_height[i],
                    vec_width[i],
                    vec_stride[i],
                )
            );
        }
        vec_points
    }

    pub fn fcos_target_single(
        &self,
        points:Tensor,
        gt_bboxes:Tensor,
        gt_labels:Tensor,
        regress_ranges:Tensor,
    )->Vec<Tensor>{
        let mut t_out:Vec<Tensor> = Vec::new();
        let num_points = points.size()[0];
        let num_gts = gt_labels.size()[0];
        if num_gts == 0{
            t_out.push(gt_labels.new_zeros(&[num_points], kind::INT64_CPU));
            t_out.push(gt_bboxes.new_zeros(&[num_points, 4], kind::INT64_CPU));
        }
        else {
            let mut  areas = (gt_bboxes.narrow(1,2,1)
                - gt_bboxes.narrow(1,0,1)+ 1)
                * (gt_bboxes.narrow(1,3,1)
                - gt_bboxes.narrow(1,1,1) + 1);

            areas = areas.reshape(&[1, num_gts]).repeat(&[num_points, 1]);

            let regress_ranges_size = regress_ranges.size();
            let regress_ranges_t = regress_ranges
                .reshape(&[regress_ranges_size[0],1,regress_ranges_size[1]])
                .expand(&[num_points, num_gts, 2], false);
            let gt_bboxes_size = gt_bboxes.size();

            let gt_bboxes_t = gt_bboxes
                .reshape(&[1, gt_bboxes_size[0], gt_bboxes_size[1]])
                .expand(&[num_points, num_gts, 4], false);

            let xs = points.narrow(1,0,1).expand(&[num_points, num_gts], false);
            let ys = points.narrow(1,1,1).expand(&[num_points, num_gts], false);

            let left =  xs.copy() - gt_bboxes_t.narrow(-1,0,1).squeeze_dim(-1);
            let right = gt_bboxes.narrow(-1,2,1).squeeze_dim(-1) - xs;
            let top = ys.copy() - gt_bboxes.narrow(-1,1,1).squeeze_dim(-1);
            let bottom = gt_bboxes.narrow(-1,3,1).squeeze_dim(-1) - ys;
            let mut bbox_targets = Tensor::stack(&[left, top, right, bottom], -1);

            let (mut bbox_targets_min, _) =  bbox_targets.min_dim(-1, true);
            bbox_targets_min = bbox_targets_min.squeeze_dim(-1);
            let inside_gt_bbox_mask = bbox_targets_min.gt(0);

            let (mut max_regress_distance, _) =  bbox_targets.max_dim(-1, true);
            max_regress_distance = max_regress_distance.squeeze_dim(-1);
            let mut inside_regress_range = max_regress_distance.gt_tensor(
                &regress_ranges
                    .narrow(-1,0,1)
            )*regress_ranges
                .narrow(-1,1,1)
                .gt_tensor(&max_regress_distance);


            areas=areas.masked_fill(&inside_gt_bbox_mask.eq(0),i64::MAX);
            areas=areas.masked_fill(&inside_regress_range.eq(0),i64::MAX);

            let (min_area, min_area_inds) = areas.min_dim(1, false);

            let mut labels = gt_labels.index_select(0, &min_area_inds).squeeze_dim(-1);
            labels=labels.masked_fill(&min_area.eq(i64::MAX), 0);
            let rng=Tensor::arange_start_step(0, num_points, 1, kind::FLOAT_CPU);

            let mut vec_bbox_target:Vec<Tensor> = Vec::new();
            for i in 0..num_points{
                vec_bbox_target.push(
                    bbox_targets
                        .select(0, i)
                        .index_select(0, &min_area_inds.select(0,i))
                );
            }
            bbox_targets = Tensor::cat(&vec_bbox_target, 0);

            t_out.push(labels);
            t_out.push(bbox_targets);
        }
        t_out
    }
    pub fn fcos_target(
        &self,
        vec_points:Vec<Tensor>,
        gt_bboxes:Tensor,
        gt_labels:Tensor,
        vec_regress_range:Vec<Tensor>,
    )->Vec<Vec<Tensor>>{
        assert_eq!(vec_points.len(), vec_regress_range.len());
        let num_levels = vec_points.len();
        let mut out :Vec<Vec<Tensor>> = Vec::new();
        for i in 0..num_levels{
            out.push(
                self.fcos_target_single(
                    vec_points[i].copy(),
                    gt_bboxes.copy(),
                    gt_labels.copy(),
                    vec_regress_range[i].expand_as(&vec_points[i]),
                )
            );
        }
        out
    }
}
