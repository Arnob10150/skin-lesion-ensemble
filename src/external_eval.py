"""Re-run external ISIC 2019 evaluation with the full image cohort.

The ISIC 2019 training release contains the complete HAM10000 image set.
Those 10,015 overlapping image IDs are excluded so external evaluation uses
only non-HAM images. For each backbone we retrain one HAM fold (using the
same random seed as the first notebook fold), then infer probabilities on
the independent ISIC images. Temperatures are re-fit on validation. External
results and ensemble tables are written to the results directory.
"""
import os, io, time, math, copy, json, random, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']; C2I = {c:i for i,c in enumerate(CLASSES)}
ISIC_MAP = {'MEL':'mel','NV':'nv','BCC':'bcc','AK':'akiec','SCC':'akiec','BKL':'bkl','DF':'df','VASC':'vasc'}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Rebuild cohorts from current on-disk images ---
def build_ham():
    m = pd.read_csv(os.path.join(DATA,'ham10000','raw','HAM10000_metadata.csv'))
    m['file'] = m['image_id'].apply(lambda i: os.path.join(DATA,'ham10000','images', f'{i}.jpg'))
    m = m[m['file'].apply(os.path.exists)].reset_index(drop=True)
    m['dx'] = m['dx'].str.lower(); m = m[m['dx'].isin(CLASSES)].reset_index(drop=True)
    m['y'] = m['dx'].map(C2I).astype(int); return m

def build_isic():
    gt = pd.read_csv(os.path.join(DATA,'isic2019','raw','ISIC_2019_Training_GroundTruth.csv'))
    rows = []
    for _, r in gt.iterrows():
        iid = r['image']; op = os.path.join(DATA,'isic2019','images', f'{iid}.jpg')
        if not os.path.exists(op): continue
        oh = r.iloc[1:].astype(float).values; cls = list(gt.columns[1:])[int(np.argmax(oh))]
        if cls not in ISIC_MAP: continue
        rows.append(dict(image_id=iid, file=op, dx=ISIC_MAP[cls], y=C2I[ISIC_MAP[cls]]))
    return pd.DataFrame(rows)

ham = build_ham(); isic_all = build_isic()
ham_image_ids = set(ham.image_id.astype(str))
isic_keep = ~isic_all.image_id.astype(str).isin(ham_image_ids)
isic = isic_all.loc[isic_keep].reset_index(drop=True)
print(f"HAM: {len(ham)} / lesions {ham.lesion_id.nunique()}; "
      f"ISIC independent: {len(isic)} / retrieved total: {len(isic_all)}; "
      f"excluded HAM overlap: {len(isic_all) - len(isic)}")

MEAN,STD = [0.485,0.456,0.406],[0.229,0.224,0.225]
T_TRAIN = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15), transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
T_EVAL = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(MEAN,STD)])

class DS(Dataset):
    def __init__(s, files, labels, train):
        s.files, s.labels, s.t = list(files), np.asarray(labels), T_TRAIN if train else T_EVAL
    def __len__(s): return len(s.files)
    def __getitem__(s,i): return s.t(Image.open(s.files[i]).convert('RGB')), int(s.labels[i])

def build(name, nc=7):
    if name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights='DEFAULT'); m.classifier[1] = nn.Linear(1280, nc); new = m.classifier.parameters()
    elif name == 'MobileNetV2':
        m = models.mobilenet_v2(weights='DEFAULT'); m.classifier[1] = nn.Linear(1280, nc); new = m.classifier.parameters()
    elif name == 'MobileNetV3-Small':
        m = models.mobilenet_v3_small(weights='DEFAULT'); in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, nc); new = m.classifier.parameters()
    elif name == 'ShuffleNetV2':
        m = models.shufflenet_v2_x1_0(weights='DEFAULT'); m.fc = nn.Linear(m.fc.in_features, nc); new = m.fc.parameters()
    elif name == 'ResNet-50':
        m = models.resnet50(weights='DEFAULT'); m.fc = nn.Linear(m.fc.in_features, nc); new = m.fc.parameters()
    elif name == 'DenseNet-121':
        m = models.densenet121(weights='DEFAULT'); m.classifier = nn.Linear(m.classifier.in_features, nc); new = m.classifier.parameters()
    return m, list(new)

def cb_focal(logits, y, counts, beta=0.999, gamma=2.0):
    eff = 1.0 - np.power(beta, counts); w = (1.0-beta) / np.maximum(eff, 1e-9)
    w = torch.tensor(w / w.sum() * len(w), dtype=torch.float32, device=logits.device)
    logp = F.log_softmax(logits,1); p = logp.exp(); ce = F.nll_loss(logp, y, reduction='none')
    pt = p.gather(1, y.unsqueeze(1)).squeeze(1)
    return (w[y] * (1-pt).pow(gamma) * ce).mean()

@torch.no_grad()
def predict(m, loader):
    m.eval(); P,Y,Z = [],[],[]
    for x,y in loader:
        with autocast('cuda', dtype=torch.float16, enabled=device.type=='cuda'):
            z = m(x.to(device, non_blocking=True))
        Z.append(z.float().cpu().numpy()); P.append(F.softmax(z.float(),1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(Z)

def fit_T(z, y):
    import scipy.optimize as so
    def nll(T):
        T = max(float(T),1e-3); return F.cross_entropy(torch.tensor(z/T), torch.tensor(y, dtype=torch.long)).item()
    return float(so.minimize_scalar(nll, bounds=(0.05,10), method='bounded').x)

# --- Train one fold per backbone and evaluate on the independent ISIC subset ---
df = ham.reset_index(drop=True); y = df.y.values; groups = df.lesion_id.values
folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(df, y, groups))
trv, te = folds[0]
tr, va = train_test_split(trv, test_size=0.10, stratify=y[trv], random_state=42)
cnt = np.bincount(y[tr], minlength=7)

trl = DataLoader(DS(df.file.values[tr], y[tr], True),  32, shuffle=True, num_workers=0, drop_last=True)
val = DataLoader(DS(df.file.values[va], y[va], False), 64, num_workers=0)
tel = DataLoader(DS(df.file.values[te], y[te], False), 64, num_workers=0)
isl = DataLoader(DS(isic.file.values,  isic.y.values, False), 64, num_workers=0)

BB = ['EfficientNet-B0','MobileNetV2','MobileNetV3-Small','ShuffleNetV2','ResNet-50','DenseNet-121']
EPOCHS = 10
ext_cache = {}

for name in BB:
    cache_p = os.path.join(RES, f"ext3_independent_{name.replace(' ','_')}.npz")
    if os.path.exists(cache_p):
        z_ = np.load(cache_p, allow_pickle=True)
        current_ids = isic.image_id.to_numpy(dtype=str)
        if 'image_id' not in z_ or not np.array_equal(np.asarray(z_['image_id'], dtype=str), current_ids):
            raise ValueError(f"Independent-cohort cache IDs do not match current ISIC rows: {cache_p}")
        ext_cache[name] = (z_['P'], z_['y'], z_['T'])
        if z_['image_id'].dtype.hasobject:
            np.savez(cache_p, P=z_['P'], y=z_['y'], T=z_['T'], image_id=current_ids)
        print(f"[cached] {name}"); continue
    # Convert legacy predictions made on all retrievable ISIC rows by matching
    # the established ground-truth CSV order, then retain non-HAM IDs only.
    legacy_p = os.path.join(RES, f"ext2_{name.replace(' ','_')}.npz")
    if os.path.exists(legacy_p):
        z_ = np.load(legacy_p)
        if len(z_['P']) != len(isic_all) or not np.array_equal(z_['y'], isic_all.y.values):
            raise ValueError(f"Legacy external cache does not align with ISIC ground truth: {legacy_p}")
        P_ind = z_['P'][isic_keep.values]
        y_ind = z_['y'][isic_keep.values]
        T_ind = z_['T']
        ext_cache[name] = (P_ind, y_ind, T_ind)
        np.savez(cache_p, P=P_ind, y=y_ind, T=T_ind, image_id=isic.image_id.to_numpy(dtype=str))
        print(f"[filtered legacy cache] {name}: {len(P_ind)} independent images"); continue
    print(f"\n--- {name} ---", flush=True)
    set_seed(1000)
    m, new = build(name); m.to(device)
    old = [p for p in m.parameters() if id(p) not in {id(x) for x in new}]
    opt = torch.optim.AdamW([{'params':old,'lr':1e-4},{'params':new,'lr':1e-3}], weight_decay=1e-2)
    steps = EPOCHS * len(trl); warm = len(trl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s<warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm))))
    scaler = GradScaler(enabled=device.type=='cuda')
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        m.train(); t0 = time.time(); tl=0; n=0
        for x,yy in trl:
            x = x.to(device, non_blocking=True); yy = yy.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.float16, enabled=device.type=='cuda'):
                z = m(x); loss = cb_focal(z, yy, cnt)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(),1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()*len(yy); n += len(yy)
        p, yv, _ = predict(m, val); f1 = f1_score(yv, p.argmax(1), average='macro')
        if f1 > best: best, bad, best_state = f1, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= 3: break
        print(f"  ep{ep+1:02d} loss {tl/n:.3f} valF1 {f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    m.load_state_dict(best_state)

    # calibrated probs on ISIC
    p_va, y_va, z_va = predict(m, val); T = fit_T(z_va, y_va)
    _, _, z_isic = predict(m, isl)
    z_cal = z_isic / T; p_isic = np.exp(z_cal - z_cal.max(1,keepdims=True)); p_isic = p_isic/p_isic.sum(1,keepdims=True)
    ext_cache[name] = (p_isic, isic.y.values, T)
    np.savez(cache_p, P=p_isic, y=isic.y.values, T=T, image_id=isic.image_id.to_numpy(dtype=str))
    del m; torch.cuda.empty_cache()

# ---- External table + ensemble ----
val_f1_sum = json.load(open(os.path.join(RES,'summary.json')))['backbones']
tau = 10.0
w_wv = {n: math.exp(tau * val_f1_sum[n]['ValMacroF1']) for n in BB}
s = sum(w_wv.values()); w_wv = {n: v/s for n,v in w_wv.items()}
POSTER3 = ['EfficientNet-B0','MobileNetV2','MobileNetV3-Small']

def soft(probs): return np.stack(list(probs.values()),0).mean(0)
def wv(probs, w):
    arr = np.stack([probs[k] for k in w.keys()],0); ww = np.array([w[k] for k in w.keys()]); ww = ww/ww.sum()
    return (arr * ww[:,None,None]).sum(0)

results_ext = {}
for n in BB: results_ext[n] = ext_cache[n][0]
ENS = {
    'Ensemble 3 (poster, soft)': soft({n: ext_cache[n][0] for n in POSTER3}),
    'Ensemble 3 (poster, weighted)': wv({n: ext_cache[n][0] for n in POSTER3}, {n:w_wv[n] for n in POSTER3}),
    'Ensemble 6 (all, soft)': soft({n: ext_cache[n][0] for n in BB}),
    'Ensemble 6 (all, weighted)': wv({n: ext_cache[n][0] for n in BB}, w_wv),
}
y_ext = isic.y.values

def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred==y).astype(float); e=0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1], np.linspace(0,1,bins+1)[1:]):
        m = (conf>lo) & (conf<=hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def metrics(P, y, nc=7):
    pred = P.argmax(1); k = min(3,nc)
    try: auc = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(nc)))
    except Exception: auc = np.nan
    return {'Accuracy': accuracy_score(y,pred), 'Balanced acc.': balanced_accuracy_score(y,pred),
            'Macro-F1': f1_score(y,pred, average='macro', labels=list(range(nc)), zero_division=0),
            'Top-3 acc.': float(np.mean([y[i] in np.argsort(P[i])[-k:] for i in range(len(y))])),
            'Kappa': cohen_kappa_score(y,pred), 'MCC': matthews_corrcoef(y,pred),
            'AUROC': auc, 'ECE': ece(P,y)}

_BOOT_WEIGHTS = None

def boot_ci(P, y, nc=7, B=2000):
    # Resample image counts once, then reuse those same image-level bootstrap
    # replicates for every model. Weighted ranks give the same one-vs-rest AUC
    # as expanding the resampled images, while avoiding repeated score sorts.
    global _BOOT_WEIGHTS
    if _BOOT_WEIGHTS is None or _BOOT_WEIGHTS.shape != (B, len(y)):
        r = np.random.RandomState(7)
        _BOOT_WEIGHTS = np.stack([
            np.bincount(r.randint(0, len(y), len(y)), minlength=len(y))
            for _ in range(B)
        ]).astype(np.int32, copy=False)

    pred = P.argmax(1)
    row_id = nc * y + pred
    auc_order = [np.argsort(P[:, k], kind='mergesort') for k in range(nc)]
    auc_starts = [
        np.r_[0, np.flatnonzero(np.diff(P[order, k]) != 0) + 1]
        for k, order in enumerate(auc_order)
    ]
    auc_is_pos = [y[order] == k for k, order in enumerate(auc_order)]
    keys = ('Accuracy','Macro-F1','AUROC','Balanced acc.')
    vals = {k: np.empty(B, dtype=float) for k in keys}

    for b, w in enumerate(_BOOT_WEIGHTS):
        cm = np.bincount(row_id, weights=w, minlength=nc*nc).reshape(nc, nc)
        tp = np.diag(cm); support = cm.sum(1); predicted = cm.sum(0)
        denom = support + predicted
        f1 = np.divide(2*tp, denom, out=np.zeros(nc), where=denom > 0)
        recall = np.divide(tp, support, out=np.zeros(nc), where=support > 0)
        vals['Accuracy'][b] = tp.sum() / w.sum()
        vals['Macro-F1'][b] = f1.mean()
        vals['Balanced acc.'][b] = recall[support > 0].mean()

        aucs = []
        for k, (order, starts, is_pos) in enumerate(zip(auc_order, auc_starts, auc_is_pos)):
            sorted_w = w[order]
            pos_w = np.add.reduceat(sorted_w * is_pos, starts)
            neg_w = np.add.reduceat(sorted_w * ~is_pos, starts)
            n_pos = pos_w.sum(); n_neg = neg_w.sum()
            if n_pos == 0 or n_neg == 0:
                aucs = []
                break
            lower_neg = np.cumsum(neg_w) - neg_w
            aucs.append(np.sum(pos_w * (lower_neg + 0.5 * neg_w)) / (n_pos * n_neg))
        vals['AUROC'][b] = np.mean(aucs) if aucs else np.nan

    return {k:(np.nanpercentile(v,2.5), np.nanpercentile(v,97.5)) for k,v in vals.items()}

def fmt(v,ci,d=3): return f'{v:.{d}f} ({ci[0]:.{d}f}-{ci[1]:.{d}f})'
def row(name, P, y_):
    m = metrics(P, y_); ci = boot_ci(P, y_)
    return {'Model': name, 'Accuracy': fmt(m['Accuracy'], ci['Accuracy']),
            'Balanced acc.': fmt(m['Balanced acc.'], ci['Balanced acc.']),
            'Macro-F1': fmt(m['Macro-F1'], ci['Macro-F1']),
            'AUROC': fmt(m['AUROC'], ci['AUROC']),
            'Top-3 acc.': f'{m["Top-3 acc."]:.3f}',
            'Kappa': f'{m["Kappa"]:.3f}', 'MCC': f'{m["MCC"]:.3f}', 'ECE': f'{m["ECE"]:.3f}'}

rows = []
for n in BB: rows.append(row(n, results_ext[n], y_ext))
for n, P in ENS.items(): rows.append(row(n, P, y_ext))
T2 = pd.DataFrame(rows); T2.to_csv(os.path.join(RES,'table_external.csv'), index=False)
print("\nTABLE 2 – External zero-shot (n={}):".format(len(y_ext)))
print(T2.to_string(index=False))
