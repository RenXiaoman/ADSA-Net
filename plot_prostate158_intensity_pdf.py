import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns


MODALITY_MAP = {
    "0000": "T2W",
    "0001": "ADC",
    "0002": "DWI",
}


def strip_nii_gz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-modality intensity PDFs for csPCa vs non-malignant tissue "
            "using Prostate158 nnUNet_train data."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="dataset/Prostate158/nnUNet_train",
        help="Root directory of the Prostate158 nnUNet_train dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/prostate158_intensity_pdf",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=200000,
        help="Maximum number of voxels sampled per modality/class for plotting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for voxel subsampling.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=-1e9,
        help="Foreground threshold; voxels <= this value are treated as background.",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=(0.5, 99.5),
        metavar=("LOW", "HIGH"),
        help="Percentile range used to clip extreme values before plotting.",
    )
    return parser.parse_args()


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def sample_array(arr: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if arr.size <= max_samples:
        return arr
    indices = rng.choice(arr.size, size=max_samples, replace=False)
    return arr[indices]


def collect_case_paths(root: Path):
    image_dirs = [root / "imagesTr", root / "imagesTs"]
    label_dirs = [root / "labelsTr", root / "labelsTs"]

    label_map = {}
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.nii.gz"):
            label_map[strip_nii_gz(label_path.name)] = label_path

    image_paths = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        image_paths.extend(sorted(image_dir.glob("*.nii.gz")))

    return image_paths, label_map


def extract_case_id(image_path: Path) -> tuple[str, str]:
    name = strip_nii_gz(image_path.name)
    case_id, modality_code = name.rsplit("_", 1)
    return case_id, modality_code


def collect_intensity_values(root: Path, fg_threshold: float):
    image_paths, label_map = collect_case_paths(root)
    data = {name: {"csPCa": [], "non_malignant": []} for name in MODALITY_MAP.values()}

    missing_labels = []
    matched_cases = set()
    for image_path in image_paths:
        case_id, modality_code = extract_case_id(image_path)
        modality = MODALITY_MAP.get(modality_code)
        if modality is None:
            continue

        label_path = label_map.get(case_id)
        if label_path is None:
            missing_labels.append(case_id)
            continue

        matched_cases.add(case_id)
        image = load_nifti(image_path)
        label = load_nifti(label_path)

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch: {image_path} has shape {image.shape}, "
                f"but {label_path} has shape {label.shape}."
            )

        finite_mask = np.isfinite(image)
        foreground_mask = finite_mask & (image > fg_threshold)
        tumor_mask = foreground_mask & (label > 0)
        non_malignant_mask = foreground_mask & (label == 0)

        tumor_values = image[tumor_mask]
        non_malignant_values = image[non_malignant_mask]

        if tumor_values.size > 0:
            data[modality]["csPCa"].append(tumor_values)
        if non_malignant_values.size > 0:
            data[modality]["non_malignant"].append(non_malignant_values)

    print(
        f"Found {len(image_paths)} image files across imagesTr and imagesTs, "
        f"covering {len(matched_cases)} matched cases."
    )
    if missing_labels:
        print(f"Warning: {len(set(missing_labels))} cases do not have labels and were skipped.")

    merged = {}
    for modality, group_data in data.items():
        merged[modality] = {}
        for group_name, arrays in group_data.items():
            if arrays:
                merged[modality][group_name] = np.concatenate(arrays, axis=0)
            else:
                merged[modality][group_name] = np.array([], dtype=np.float32)
    return merged


def clip_values(values_a: np.ndarray, values_b: np.ndarray, low_pct: float, high_pct: float):
    merged = np.concatenate([values_a, values_b], axis=0)
    low, high = np.percentile(merged, [low_pct, high_pct])
    return np.clip(values_a, low, high), np.clip(values_b, low, high), low, high


def plot_probability_hist(ax, values: np.ndarray, label: str, color: str):
    sns.histplot(
        values,
        ax=ax,
        stat="probability",
        bins=200,
        element="step",
        fill=True,
        alpha=0.25,
        linewidth=1.5,
        color=color,
        label=label,
        common_norm=False,
    )


def plot_modality_pdf(modality: str, tumor: np.ndarray, non_malignant: np.ndarray, output_dir: Path):
    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 5.5))

    plot_probability_hist(ax, non_malignant, "Non-malignant tissue", "#1f77b4")
    plot_probability_hist(ax, tumor, "csPCa", "#d62728")

    ax.set_title(f"{modality}: Intensity Distribution")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Probability")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_pdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_pdf(all_data: dict, output_dir: Path):
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=False)

    for ax, modality in zip(axes, ["T2W", "ADC", "DWI"]):
        tumor = all_data[modality]["csPCa"]
        non_malignant = all_data[modality]["non_malignant"]
        if tumor.size == 0 or non_malignant.size == 0:
            ax.set_title(f"{modality}: no valid data")
            ax.axis("off")
            continue

        plot_probability_hist(ax, non_malignant, "Non-malignant tissue", "#1f77b4")
        plot_probability_hist(ax, tumor, "csPCa", "#d62728")
        ax.set_title(modality)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Probability")
        ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "all_modalities_pdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = collect_intensity_values(root, args.foreground_threshold)

    low_pct, high_pct = args.clip_percentiles
    for modality in ["T2W", "ADC", "DWI"]:
        tumor = all_data[modality]["csPCa"]
        non_malignant = all_data[modality]["non_malignant"]
        if tumor.size == 0 or non_malignant.size == 0:
            print(f"{modality}: skipped because one class is empty.")
            continue

        tumor, non_malignant, low, high = clip_values(tumor, non_malignant, low_pct, high_pct)
        tumor = sample_array(tumor, args.max_samples_per_class, rng)
        non_malignant = sample_array(non_malignant, args.max_samples_per_class, rng)

        all_data[modality]["csPCa"] = tumor
        all_data[modality]["non_malignant"] = non_malignant

        print(
            f"{modality}: csPCa={tumor.size}, non_malignant={non_malignant.size}, "
            f"clip_range=[{low:.4f}, {high:.4f}]"
        )
        plot_modality_pdf(modality, tumor, non_malignant, output_dir)

    plot_combined_pdf(all_data, output_dir)
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
