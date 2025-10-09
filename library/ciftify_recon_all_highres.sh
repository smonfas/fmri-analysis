#!/bin/bash

# runs ciftify_recon-all on highres data, using experts settings and 0.5mm MNI templates
# that are being imported from FSL
#
# usage: ./ciftify_recon_all_highres.sh <ciftify_dir> <freesurfer_dir> <subject>
#
# The freesurfer subject directory is assume to be in <freesurfer_dir>/<subject>
# The ciftify output directory is <ciftify_dir>/<subject>
ciftify_dir=$(realpath "$1")
freesurfer_dir=$(realpath "$2")
subject=$3
standard_template_basename_path=$4


# run everything from a temporary directory
cwd=$(pwd)
tmp_dir=$(mktemp -d)
trap 'cd "${cwd}"; rm -rf ${tmp_dir}' EXIT
cd ${tmp_dir}

# if standard_template_basename_path is not provided, import and process a 0.5 mm MNI template
if [ -z "$standard_template_basename_path" ]; then
      # import from tensorflow
      tf_template_T1w_highres=$(python -c \
        "from templateflow import api as tflow; print(tflow.get('MNI152NLin6Asym', suffix='T1w', resolution=3))" 2>/dev/null || true)
      tf_template_brainmask_lowres=$(python -c \
        "from templateflow import api as tflow; print(tflow.get('MNI152NLin6Asym', desc='brain', suffix='mask', resolution=1))" 2>/dev/null || true)

      # check if tensorflow import succeeded, then copy files otherwise import from FSL instead
      if [ -f ${tf_template_T1w_highres} ] && [ -f ${tf_template_brainmask_lowres} ]; then
            cp ${tf_template_T1w_highres} MNI152_T1_0.5mm.nii.gz
            cp ${tf_template_brainmask_lowres} MNI152_T1_1mm_brain_mask.nii.gz
      else
            echo "TemplateFlow download failed, using local fallback (import from fsl)..."
            export FSLOUTPUTTYPE=NIFTI_GZ
            cp ${FSL_DIR}/data/standard/MNI152_T1_0.5mm.nii.gz MNI152_T1_0.5mm.nii.gz
            cp ${FSL_DIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz MNI152_T1_1mm_brain_mask.nii.gz
      fi
      # create a 0.5mm brain mask
      flirt -in MNI152_T1_1mm_brain_mask.nii.gz \
            -ref MNI152_T1_0.5mm.nii.gz \
            -out MNI152_T1_0.5mm_brain_mask.nii.gz \
            -applyxfm \
            -interp nearestneighbour

      fslmaths MNI152_T1_0.5mm.nii.gz -mas MNI152_T1_0.5mm_brain_mask.nii.gz \
            MNI152_T1_0.5mm_brain.nii.gz

      standard_template_basename_path=${tmp_dir}/MNI152_T1_0.5mm_brain
fi

# create expert settings file by writing the following content to ciftify_expert_settings_0.5mm.yaml
cat << EOF > ciftify_expert_settings_0.5mm.yaml
high_res : "164"
low_res : ["164"]
grayord_res : [2]

# Ensure 'mask_medialwall' is a boolean value, NOT string. i.e. False and not 'False'
dscalars : {

        sulc : {
              mapname: sulc,
              fsname: sulc,
              map_postfix: _Sulc,
              palette_mode: MODE_AUTO_SCALE_PERCENTAGE,
              palette_options: -pos-percent 2 98 -palette-name Gray_Interp -disp-pos true -disp-neg true -disp-zero true,
              mask_medialwall: False
              },

        curvature : {
              mapname: curvature,
              fsname: curv,
              map_postfix: _Curvature,
              palette_mode: MODE_AUTO_SCALE_PERCENTAGE,
              palette_options: -pos-percent 2 98 -palette-name Gray_Interp -disp-pos true -disp-neg true -disp-zero true,
              mask_medialwall: True
              },

        thickness : {
              mapname: thickness,
              fsname: thickness,
              map_postfix: _Thickness,
              palette_mode: MODE_AUTO_SCALE_PERCENTAGE,
              palette_options: -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false,
              mask_medialwall: True
              },

        ArealDistortion_FS : {
              mapname : ArealDistortion_FS,
              map_postfix: _ArealDistortion_FS,
              palette_mode: MODE_USER_SCALE,
              palette_options: -pos-user 0 1 -neg-user 0 -1 -interpolate true -palette-name ROY-BIG-BL -disp-pos true -disp-neg true -disp-zero false,
              mask_medialwall: False
              },

        ArealDistortion_MSMSulc : {
              mapname: ArealDistortion_MSMSulc,
              map_postfix: _ArealDistortion_MSMSulc,
              palette_mode: MODE_USER_SCALE,
              palette_options: -pos-user 0 1 -neg-user 0 -1 -interpolate true -palette-name ROY-BIG-BL -disp-pos true -disp-neg true -disp-zero false,
              mask_medialwall: False
              },

        EdgeDistortion_MSMSulc : {
              mapname: EdgeDistortion_MSMSulc,
              map_postfix: _EdgeDistortion_MSMSulc,
              palette_mode: MODE_USER_SCALE,
              palette_options: -pos-user 0 1 -neg-user 0 -1 -interpolate true -palette-name ROY-BIG-BL -disp-pos true -disp-neg true -disp-zero false,
              mask_medialwall: False
              }
}

registration : {
        src_mesh: T1wNative,
        dest_mesh: AtlasSpaceNative,
        src_dir: T1w,
        dest_dir: MNINonLinear,
        xfms_dir : MNINonLinear/xfms,
        T1wImage : T1w.nii.gz,
        T1wBrain : T1w_brain.nii.gz,
        BrainMask : brainmask_fs.nii.gz,
        AtlasTransform_Linear : T1w2StandardLinear.mat,
        AtlasTransform_NonLinear : T1w2Standard_warp_noaffine.nii.gz,
        InverseAtlasTransform_NonLinear : Standard2T1w_warp_noaffine.nii.gz
}

# Define registration methods dictionaries here, with the available resolutions for each
# defined inside
FSL_fnirt : {
      2mm : {
            FNIRTConfig : etc/flirtsch/T1_2_MNI152_2mm.cnf,
            standard_T1wImage : ${standard_template_basename_path}.nii.gz,
            standard_BrainMask: ${standard_template_basename_path}_brain_mask.nii.gz,
            standard_T1wBrain : ${standard_template_basename_path}_brain.nii.gz
        }
}
EOF
cat ciftify_expert_settings_0.5mm.yaml

ciftify_recon_all --ciftify-work-dir ${ciftify_dir} \
               --ciftify-conf ciftify_expert_settings_0.5mm.yaml \
               --resample-to-T1w32k \
               --fs-subjects-dir ${freesurfer_dir} ${subject}