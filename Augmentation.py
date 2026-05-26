import numpy as np
import torchio as tio
import os
import nibabel as nib

# ====== # 
# CONFIG # 
# ====== # 

CONFIG = {
    "input_dir": "/data/dataset/",
    "output_dir": "/data/aug_dataset/",
    "seed": 123,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ===================== # 
# AUGMENTATION PIPELINE # 
# ===================== # 

def get_transform():

    return tio.OneOf({
        tio.RandomElasticDeformation(
            num_control_points=8,
            max_displacement=4.0
        ): 1.0,

        tio.RandomGamma(
            log_gamma=(-0.1, 0.1)
        ): 1.0,

        tio.RandomBiasField(
            coefficients=0.1
        ): 1.0,
    })


transform = get_transform()


# ===================== # 
# AUGMENTATION FUNCTION # 
# ===================== # 

def augment_nifti_file(filepath, transform, output_dir):

    img = nib.load(filepath)

    data = img.get_fdata(dtype=np.float32)  
    affine = img.affine

    subject = tio.Subject(
        mri=tio.ScalarImage(
            tensor=data[np.newaxis, ...],
            affine=affine
        )
    )

    augmented = transform(subject)

    aug_data = augmented["mri"].data[0].numpy()

    filename = os.path.basename(filepath)
    out_path = os.path.join(output_dir, f"aug_{filename}")

    nib.save(
        nib.Nifti1Image(aug_data, affine),
        out_path
    )

    return filename


# ========= # 
# MAIN LOOP # 
# ========= # 

def run_augmentation(config, transform):

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    files = [
        f for f in os.listdir(input_dir)
        if f.endswith((".nii", ".nii.gz"))
    ]

    print(f"Found {len(files)} MRI files")

    for fname in files:

        in_path = os.path.join(input_dir, fname)

        try:
            augment_nifti_file(in_path, transform, output_dir)
            print(f"{fname} → augmented")

        except Exception as e:
            print(f"Failed {fname}: {e}")


# === # 
# RUN # 
# === # 

if __name__ == "__main__":
    run_augmentation(CONFIG, transform)