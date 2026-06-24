"""
SignTax — Advertising Sign size estimator (Streamlit app)

Wraps the exact pipeline from
`SignTax_Depth_SizeEstimation take out resize in proprocessing.ipynb`:

    load image -> Faster R-CNN (ResNet101-FPN) detect -> SCORE_THRESH filter
    -> class-agnostic NMS -> (optional) top-1 -> Depth Pro depth map + focal length
    -> per box: Z = median depth of central region
    -> RANSAC plane-fit inside the box -> surface normal -> foreshortening correction
       (1/cos of the tilt, separately for width & height) -> W = w_px * Z / f * corr * K

Run:
    depth-pro\\venv\\Scripts\\python.exe -m streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --------------------------------------------------------------------------- #
# Paths / constants (mirror the notebook's Config cell)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent
DEPTH_PRO_DIR = ROOT / "depth-pro"
DEPTH_CKPT_DEFAULT = DEPTH_PRO_DIR / "checkpoints" / "depth_pro.pt"
DETECTOR_CKPT_DEFAULT = ROOT / "Object-Detection-And-Size-Estimation" / "fasterrcnn_resnet101_head.pth"

# Weights are hosted on the Hugging Face Hub so they don't bloat the git repo.
# When the local checkpoint above is missing (e.g. on Spaces), it's pulled from here.
HF_WEIGHTS_REPO = "prannanan/SignTax"
DEPTH_HF_FILE = "depth_pro.pt"
DETECTOR_HF_FILE = "fasterrcnn_resnet101_finetuned_no_resize.pth"

CLASS_NAMES = ["__background__", "sign"]   # must match training order (2-class checkpoint)
TARGET_W, TARGET_H = 3024, 4032            # camera resolution the boxes/depth are aligned to
CALIB_SCALE = 1.2404                       # K — size-correction factor fitted in the notebook (## 8b)
TILT_BAND = 0.5                            # metres: depth window around the box centre used to
                                           #     drop sky/background before RANSAC plane-fitting
MAX_CORR = 3.0                             # cap on the 1/cos foreshortening factor (guards against
                                           #     blow-up when the tilt estimate is near 90 degrees)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
DEPTH_PRECISION = torch.half if DEVICE.type == "cuda" else torch.float32

# Make the local depth_pro package importable (same trick as the notebook).
_src = (DEPTH_PRO_DIR / "src").resolve()
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from depth_pro import create_model_and_transforms, load_rgb            # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT          # noqa: E402
from torchvision.models.detection import FasterRCNN                    # noqa: E402
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # noqa: E402
from torchvision.ops import nms                                        # noqa: E402


# --------------------------------------------------------------------------- #
# Model loading (cached so it happens once per process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading Depth Pro (~1.9 GB checkpoint) ...")
def load_depth_model(depth_ckpt: str):
    cfg = DEFAULT_MONODEPTH_CONFIG_DICT
    cfg.checkpoint_uri = str(Path(depth_ckpt).resolve())
    model, transform = create_model_and_transforms(
        config=cfg, device=DEVICE, precision=DEPTH_PRECISION
    )
    model.eval()
    return model, transform


@st.cache_resource(show_spinner="Loading Faster R-CNN detector ...")
def load_detector(detector_ckpt: str):
    backbone = resnet_fpn_backbone(backbone_name="resnet101", weights=None, trainable_layers=0)
    model = FasterRCNN(backbone, num_classes=91)            # COCO head, matches training before swap
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES))
    state = torch.load(detector_ckpt, map_location=DEVICE)
    # Accept either a raw state_dict or a checkpoint dict that wraps it.
    if isinstance(state, dict) and "model" in state and "roi_heads.box_predictor.cls_score.weight" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner="Downloading model weights from Hugging Face ...")
def fetch_weight(filename: str) -> str:
    """Download a checkpoint from the HF Hub once; return its local cache path."""
    return hf_hub_download(repo_id=HF_WEIGHTS_REPO, filename=filename)


def resolve_ckpt(path_str: str, hf_filename: str) -> str:
    """Use the local checkpoint if present, otherwise pull it from the HF Hub."""
    return path_str if Path(path_str).exists() else fetch_weight(hf_filename)


def load_upright(path) -> Image.Image:
    """Open an image and apply its EXIF orientation so phone photos aren't sideways."""
    return ImageOps.exif_transpose(Image.open(path))


# --------------------------------------------------------------------------- #
# Tilt correction — RANSAC plane-fit on the depth inside a box, then the
# foreshortening factors 1/cos(tilt) for width and height (mirrors the
# TiltSign notebook).
# --------------------------------------------------------------------------- #
def _backproject_box(depth_map, focal_px, box):
    """Pixels inside `box` -> 3D points (N,3) + their depth (N,), via pinhole."""
    h_img, w_img = depth_map.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    cx, cy = w_img / 2.0, h_img / 2.0
    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    Z = depth_map[y1:y2, x1:x2]
    X = (us - cx) * Z / focal_px
    Y = (vs - cy) * Z / focal_px
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts, Z.reshape(-1)


def _ransac_normal(P, thresh=0.02, iters=200, seed=0):
    """Robustly fit a plane to 3D points and return its unit normal (or None)."""
    if len(P) < 50:
        return None
    rng = np.random.default_rng(seed)
    Ps = P if len(P) <= 20000 else P[rng.choice(len(P), 20000, replace=False)]
    best_cnt, best_n = -1, None
    for _ in range(iters):
        a, b, c = Ps[rng.choice(len(Ps), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        cnt = int((np.abs((Ps - a) @ nrm) < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best_n = cnt, nrm
    return best_n


def box_tilt_correction(depth_map, focal_px, box, Z):
    """RANSAC plane-fit inside the box -> (corr_w, corr_h, tilt_deg).

    Returns 1/cos of the per-axis tilt so width/height can be un-foreshortened.
    Falls back to (1, 1, 0) when the depth is too sparse/noisy to fit a plane.
    """
    if not np.isfinite(Z):
        return 1.0, 1.0, 0.0
    pts, depth = _backproject_box(depth_map, focal_px, box)
    keep = np.abs(depth - Z) < TILT_BAND            # drop sky/background by depth
    pts = pts[keep] if keep.sum() >= 50 else pts
    n = _ransac_normal(pts)
    if n is None:
        return 1.0, 1.0, 0.0
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    eps = 1e-6
    corr_w = min(np.sqrt(nx * nx + nz * nz) / (abs(nz) + eps), MAX_CORR)
    corr_h = min(np.sqrt(ny * ny + nz * nz) / (abs(nz) + eps), MAX_CORR)
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(nz)))))
    return float(corr_w), float(corr_h), tilt_deg


# --------------------------------------------------------------------------- #
# Pipeline (detector + Depth Pro + tilt correction, matching the notebook)
# --------------------------------------------------------------------------- #
def run_pipeline(image_path, detector, depth_model, depth_transform,
                 score_thresh, nms_iou, center_frac, calib_scale, top1):
    # 1. Load (EXIF-rotated) and force the camera resolution so pixels line up.
    image_np, _, f_px_exif = load_rgb(str(image_path))
    if (image_np.shape[1], image_np.shape[0]) != (TARGET_W, TARGET_H):
        image_np = np.asarray(
            Image.fromarray(image_np).resize((TARGET_W, TARGET_H), Image.BICUBIC)
        )
    H, W = image_np.shape[:2]

    # 2. Detect.
    img_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float().div(255.0).to(DEVICE)
    with torch.no_grad():
        det = detector([img_tensor])[0]
    keep = det["scores"].cpu().numpy() >= score_thresh
    boxes = det["boxes"].cpu().numpy()[keep]
    scores = det["scores"].cpu().numpy()[keep]
    labels = det["labels"].cpu().numpy()[keep]

    # 3. Class-agnostic NMS.
    if len(boxes):
        k = nms(torch.from_numpy(boxes).float(),
                torch.from_numpy(scores).float(), nms_iou).numpy()
        boxes, scores, labels = boxes[k], scores[k], labels[k]

    # 4. Optional top-1 (single most-confident sign).
    if top1 and len(boxes):
        b = int(np.argmax(scores))
        boxes, scores, labels = boxes[b:b + 1], scores[b:b + 1], labels[b:b + 1]

    # 5. Depth + focal length (same numpy array as the detector).
    with torch.no_grad():
        pred = depth_model.infer(depth_transform(image_np), f_px=f_px_exif)
    depth_map = pred["depth"].detach().cpu().numpy().squeeze()
    _fl = pred["focallength_px"]
    focal_px = float(_fl.detach().cpu().item()) if torch.is_tensor(_fl) else float(_fl)

    # 6. Per-box distance, tilt correction, and real-world size.
    results = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        hw, hh = (x2 - x1) * center_frac / 2.0, (y2 - y1) * center_frac / 2.0
        rx1, ry1 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
        rx2, ry2 = min(W, int(round(cx + hw))), min(H, int(round(cy + hh)))
        region = depth_map[ry1:ry2, rx1:rx2]
        Z = float(np.median(region)) if region.size else float("nan")

        # Tilt correction: RANSAC plane-fit inside the box -> foreshortening factors.
        corr_w, corr_h, tilt_deg = box_tilt_correction(depth_map, focal_px, box, Z)

        w_px, h_px = float(x2 - x1), float(y2 - y1)
        raw_w = w_px * Z / focal_px * corr_w          # tilt-corrected (before calibration)
        raw_h = h_px * Z / focal_px * corr_h
        name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"class_{int(label)}"
        results.append({
            "class": name,
            "score": float(score),
            "box": [float(v) for v in box],
            "distance_m": Z,
            "tilt_deg": tilt_deg,
            "width_m": raw_w * calib_scale,
            "height_m": raw_h * calib_scale,
            "width_m_raw": raw_w,
            "height_m_raw": raw_h,
        })

    return image_np, depth_map, focal_px, f_px_exif, results


def render_detections(image_np, results):
    """Original photo with green boxes + per-sign size labels."""
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(image_np)
    ax.axis("off")
    for r in results:
        x1, y1, x2, y2 = r["box"]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2.5, edgecolor="#1FCB6B", facecolor="none"))
        ax.text(x1, max(0, y1 - 8),
                f"{r['class']} {r['score']:.2f}\n"
                f"{r['distance_m']:.1f} m  ·  tilt {r.get('tilt_deg', 0):.0f}°  ·  "
                f"{r['width_m']:.2f}×{r['height_m']:.2f} m",
                color="#0B2E1A", fontsize=10, va="bottom", weight="bold",
                bbox=dict(facecolor="#D7FFE6", alpha=0.92, edgecolor="none",
                          boxstyle="round,pad=0.35"))
    fig.tight_layout(pad=0.2)
    return fig


def render_depth(depth_map, results):
    """Depth Pro inverse-depth heatmap with the same boxes overlaid."""
    fig, ax = plt.subplots(figsize=(9, 12))
    inv = 1.0 / np.clip(depth_map, 1e-6, None)
    lo, hi = max(1 / 250, inv.min()), min(inv.max(), 1 / 0.1)
    ax.imshow((inv - lo) / (hi - lo + 1e-9), cmap="turbo")
    ax.axis("off")
    for r in results:
        x1, y1, x2, y2 = r["box"]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor="white", facecolor="none"))
    fig.tight_layout(pad=0.2)
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SignTax — Sign Size Estimator",
    page_icon="🪧",
    layout="wide",
)

# --- light cosmetic polish (cards, spacing, sidebar tint) ------------------ #
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1400px;}
      section[data-testid="stSidebar"] {background: #F7FAFD;}
      [data-testid="stMetric"] {
          background: #F2F6FB; border: 1px solid #E1E9F3;
          border-radius: 14px; padding: 14px 18px;
      }
      [data-testid="stMetricLabel"] p {font-size: 0.8rem; opacity: 0.65;}
      .device-pill {
          display: inline-block; padding: 2px 12px; border-radius: 999px;
          font-size: 0.82rem; font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- header ---------------------------------------------------------------- #
_dev = str(DEVICE)
_pill = "#1FA463" if DEVICE.type in ("cuda", "mps") else "#8A98AB"
st.markdown("# 🪧 SignTax — Advertising Sign Size Estimator")
st.markdown(
    f"<span style='color:#5B6B7F'>Faster R-CNN (ResNet101-FPN) + Apple Depth Pro</span>"
    f"&nbsp;&nbsp;<span class='device-pill' style='background:{_pill}22;color:{_pill}'>"
    f"● {_dev}</span>",
    unsafe_allow_html=True,
)
st.write("")

# --- sidebar --------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("📁 Model checkpoints", expanded=False):
        detector_ckpt = st.text_input(
            "Detector (.pth)", str(DETECTOR_CKPT_DEFAULT),
            help="Faster R-CNN ResNet101-FPN weights.",
        )
        depth_ckpt = st.text_input(
            "Depth Pro (.pt)", str(DEPTH_CKPT_DEFAULT),
            help="Apple Depth Pro checkpoint (~1.9 GB).",
        )

    st.subheader("🎯 Detection")
    score_thresh = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
        help="`SCORE_THRESH` — keep boxes scoring at least this.",
    )
    nms_iou = st.slider(
        "NMS overlap (IoU)", 0.0, 1.0, 0.3, 0.05,
        help="`NMS_IOU_THRESH` — merge boxes overlapping more than this.",
    )
    top1 = st.toggle(
        "Best sign only (top-1)", value=True,
        help="Keep only the single most-confident detection per image.",
    )

    st.subheader("📐 Measurement")
    center_frac = st.slider(
        "Depth sampling region", 0.1, 1.0, 0.5, 0.05,
        help="`CENTER_FRACTION` — central fraction of each box used for depth.",
    )
    st.caption(f"Calibration factor K = **{CALIB_SCALE:.4f}** (fixed)")

try:
    detector_ckpt = resolve_ckpt(detector_ckpt, DETECTOR_HF_FILE)
    depth_ckpt = resolve_ckpt(depth_ckpt, DEPTH_HF_FILE)
except Exception as exc:  # noqa: BLE001
    st.error(
        f"Could not obtain model weights from Hugging Face (`{HF_WEIGHTS_REPO}`):\n\n{exc}"
    )
    st.stop()

# --- input ----------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Upload a sign photo", type=["jpg", "jpeg", "png", "bmp"],
    help="The image is aligned to the camera resolution before measuring.",
)

if uploaded is None:
    st.info("⬆️ Upload a photo to estimate sign size. Tune thresholds in the sidebar.")

    # --- photo guidance (do / don't examples) ------------------------------ #
    st.markdown("##### 📸 ตัวอย่างการถ่ายรูป (Photo tips)")
    g_do, g_dont = st.columns(2)
    with g_do:
        with st.container(border=True):
            st.markdown(
                "<span style='background:#1FA46322;color:#1FA463;font-weight:700;"
                "padding:2px 12px;border-radius:999px;'>✅ ควรทำ (Do)</span>",
                unsafe_allow_html=True,
            )
            st.image(load_upright(ROOT / "assets" / "example_do.jpg"), width=240)
            st.caption(
                "รูปป้ายจะต้องอยู่กึ่งกลางเฟรม ไม่ถ่ายเอียง "
                "ต้องเห็นสภาพแวดล้อมภายนอก เช่น เสาไฟ หรือรถ หรือถนน"
            )
    with g_dont:
        with st.container(border=True):
            st.markdown(
                "<span style='background:#E5484D22;color:#E5484D;font-weight:700;"
                "padding:2px 12px;border-radius:999px;'>❌ ไม่ควรทำ (Don't)</span>",
                unsafe_allow_html=True,
            )
            st.image(load_upright(ROOT / "assets" / "example_dont.jpg"), width=240)
            st.caption(
                "พื้นหลังจะต้องไม่เป็นพื้นหลังที่เป็นกำแพงอย่างเดียว "
                "หรือเป็นรูปที่ไม่เห็นสภาพแวดล้อมภายนอก"
            )

    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. **Detect** signs with Faster R-CNN, filter by confidence, merge overlaps (NMS).\n"
            "2. **Estimate depth** with Apple Depth Pro → distance *Z* + focal length *f*.\n"
            "3. **Measure tilt**: RANSAC plane-fit on the depth inside each box → surface normal "
            "→ how many degrees the sign is angled away from the camera.\n"
            "4. **Measure size**: real size = box pixels × *Z* / *f* × *(1/cos tilt)* × *K* — the "
            "pinhole camera model plus a foreshortening correction so tilted signs aren't underestimated."
        )
    st.stop()

# --- run pipeline ---------------------------------------------------------- #
detector = load_detector(detector_ckpt)
depth_model, depth_transform = load_depth_model(depth_ckpt)

# load_rgb needs a path, so persist the upload to a temp file.
suffix = Path(uploaded.name).suffix or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

with st.spinner("Running detection + depth (this can take a while on CPU) ..."):
    image_np, depth_map, focal_px, f_px_exif, results = run_pipeline(
        tmp_path, detector, depth_model, depth_transform,
        score_thresh, nms_iou, center_frac, CALIB_SCALE, top1,
    )

# --- headline result ------------------------------------------------------- #
# Hero card: the size of the most-confident sign, shown first so users
# immediately see what the app produces — width × height.
if results:
    primary = max(results, key=lambda r: r["score"])
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#E9FBF1,#EAF2FB);
                    border:1px solid #D7E6DC;border-radius:18px;
                    padding:22px 26px;margin-bottom:16px;">
          <div style="font-size:0.82rem;letter-spacing:.05em;text-transform:uppercase;
                      color:#5B6B7F;font-weight:700;">Estimated sign size</div>
          <div style="font-size:2.7rem;font-weight:800;color:#0B2E1A;
                      line-height:1.1;margin-top:4px;">
            {primary['width_m']:.2f} m&nbsp;&times;&nbsp;{primary['height_m']:.2f} m
          </div>
          <div style="font-size:0.92rem;color:#5B6B7F;margin-top:8px;">
            width &times; height&nbsp;&nbsp;·&nbsp;&nbsp;{primary['distance_m']:.1f} m away
            &nbsp;&nbsp;·&nbsp;&nbsp;{primary['class']} ({primary['score']:.0%} confidence)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if len(results) > 1:
        st.caption(
            f"Showing the most confident of {len(results)} detected signs — "
            "see the full table below."
        )

# --- summary metrics ------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Signs detected", len(results))
c2.metric("Focal length used", f"{focal_px:.0f} px")
c3.metric("EXIF focal", "—" if f_px_exif is None else f"{f_px_exif:.0f} px")
if results:
    _nearest = min(results, key=lambda r: r["distance_m"])
    c4.metric("Nearest sign", f"{_nearest['distance_m']:.1f} m",
              help=f"{_nearest['width_m']:.2f} × {_nearest['height_m']:.2f} m")
else:
    c4.metric("Nearest sign", "—")

# --- measurements ---------------------------------------------------------- #
if results:
    st.subheader("📊 All measurements")
    df = pd.DataFrame([{
        "Class": r["class"],
        "Score": round(r["score"], 3),
        "Distance (m)": round(r["distance_m"], 2),
        "Tilt (°)": round(r.get("tilt_deg", 0.0), 0),
        "Width (m)": round(r["width_m"], 2),
        "Height (m)": round(r["height_m"], 2),
    } for r in results])

    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Distance (m)": st.column_config.NumberColumn(format="%.2f m"),
            "Tilt (°)": st.column_config.NumberColumn(format="%.0f°"),
            "Width (m)": st.column_config.NumberColumn(format="%.2f m"),
            "Height (m)": st.column_config.NumberColumn(format="%.2f m"),
        },
    )
    st.download_button(
        "⬇️ Download measurements (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"signtax_{Path(uploaded.name).stem}.csv",
        mime="text/csv",
    )
    st.caption("**Width/Height** are tilt-corrected (foreshortening) and include the calibration factor K.")
else:
    st.warning("No detections above the score threshold — try lowering the confidence slider.")

# --- visual output (tabs) -------------------------------------------------- #
tab_det, tab_depth = st.tabs(["📷 Detections", "🗺️ Depth map"])
with tab_det:
    st.pyplot(render_detections(image_np, results), use_container_width=True)
with tab_depth:
    st.pyplot(render_depth(depth_map, results), use_container_width=True)
    st.caption("Inverse-depth heatmap — warmer (red) is closer, cooler (blue) is farther.")
