"""Full training + internal evaluation pipeline (extracted from the notebook).

Run: python src/train_eval.py  (set CF_SMOKE=1 for a fast end-to-end check).
"""
# ===== Cell 1 =====
# ===== Cell 1 — Environment =====
import os, io, math, time, json, copy, random, hashlib, warnings, collections, glob
import concurrent.futures as cf
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, precision_recall_fscore_support)
from torch.amp import autocast, GradScaler

ROOT = os.getcwd()
DATA, RES, FIG = [os.path.join(ROOT, d) for d in ('data','results','figures')]
for d in (DATA, RES, FIG,
          os.path.join(DATA,'ham10000','images'),
          os.path.join(DATA,'isic2019','images')): os.makedirs(d, exist_ok=True)

SMOKE = os.environ.get('CF_SMOKE','0') == '1'
CFG = dict(img=224, bs=32, epochs=(1 if SMOKE else 12), patience=3, lr=1e-4,
           head_lr_mult=10, wd=1e-2, label_smooth=0.0, folds=(2 if SMOKE else 3),
           seed=42, n_boot=(50 if SMOKE else 2000), cb_beta=0.999, focal_gamma=2.0)
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG['seed'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'figure.dpi':110,'savefig.dpi':300,'font.family':'DejaVu Sans',
                     'axes.spines.top':False,'axes.spines.right':False})
print('Device:', device,
      '|', torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')
print('torch', torch.__version__, '| smoke:', SMOKE)
print(json.dumps(CFG))

# ===== Cell 3 =====
# ===== Cell 2 — Build HAM10000 + ISIC2019 cohorts from local thumbnails =====
# Images were fetched by code/fetch_images.py from the ISIC S3 bucket thumbnails.
CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']
C2I = {c:i for i,c in enumerate(CLASSES)}
DX_MAP = {'akiec':'akiec','bcc':'bcc','bkl':'bkl','df':'df','mel':'mel','nv':'nv','vasc':'vasc'}
ISIC_MAP = {'MEL':'mel','NV':'nv','BCC':'bcc','AK':'akiec','SCC':'akiec',
            'BKL':'bkl','DF':'df','VASC':'vasc'}

def _build_ham(meta_csv, img_dir, cache):
    if os.path.exists(cache): return pd.read_csv(cache)
    m = pd.read_csv(meta_csv)
    m['file'] = m['image_id'].apply(lambda i: os.path.join(img_dir, f'{i}.jpg'))
    m = m[m['file'].apply(os.path.exists)].reset_index(drop=True)
    m['dx'] = m['dx'].str.lower()
    m = m[m['dx'].isin(CLASSES)].reset_index(drop=True)
    m['y'] = m['dx'].map(C2I).astype(int)
    m = m[['image_id','lesion_id','dx','y','file','age','sex','localization']]
    m.to_csv(cache, index=False); return m

def _build_isic(gt_csv, img_dir, cache):
    ham_ids = set(pd.read_csv(os.path.join(DATA,'ham10000','raw','HAM10000_metadata.csv'))['image_id'].astype(str))
    metadata_path = os.path.join(DATA,'isic2019','raw','ISIC_2019_Training_Metadata.csv')
    isic_meta = pd.read_csv(metadata_path).set_index('image')['lesion_id'].to_dict()
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        df = df[~df.image_id.astype(str).isin(ham_ids)].reset_index(drop=True)
        df.to_csv(cache, index=False)
        return df
    gt = pd.read_csv(gt_csv)
    rows = []
    for _, r in gt.iterrows():
        iid = r['image']; op = os.path.join(img_dir, f'{iid}.jpg')
        if not os.path.exists(op) or str(iid) in ham_ids: continue
        oh = r.iloc[1:].astype(float).values; cols = list(gt.columns[1:])
        cls = cols[int(np.argmax(oh))]
        if cls not in ISIC_MAP: continue
        dx = ISIC_MAP[cls]; y = C2I[dx]
        rows.append(dict(image_id=iid, lesion_id=isic_meta.get(iid), dx=dx, y=y,
                         file=op, age=None, sex=None, localization=None))
    df = pd.DataFrame(rows); df.to_csv(cache, index=False); return df

ham = _build_ham(os.path.join(DATA,'ham10000','raw','HAM10000_metadata.csv'),
                 os.path.join(DATA,'ham10000','images'),
                 os.path.join(DATA,'ham10000','cohort.csv'))
print(f'HAM10000: {len(ham)} images, {ham.lesion_id.nunique()} unique lesions')
print(ham.dx.value_counts().reindex(CLASSES).to_string())

isic = _build_isic(os.path.join(DATA,'isic2019','raw','ISIC_2019_Training_GroundTruth.csv'),
                   os.path.join(DATA,'isic2019','images'),
                   os.path.join(DATA,'isic2019','cohort.csv'))
print(f'ISIC2019 independent non-HAM subset: {len(isic)} images')
print(isic.dx.value_counts().reindex(CLASSES).to_string() if len(isic) else '(no isic images yet)')

# ===== Cell 5 =====
# ===== Cell 3 — Cohort description =====
def desc(df, name):
    def pc(x, d): return f'{x} ({100*x/d:.1f})' if d else '—'
    n = len(df); nl = df.lesion_id.nunique()
    out = {'Images (n)': n, 'Unique lesions (n)': nl}
    if 'sex' in df.columns:
        out['Female, n (%)']  = pc((df.sex=='female').sum(), n)
        out['Male, n (%)']    = pc((df.sex=='male').sum(), n)
    if 'age' in df.columns:
        out['Age, median [IQR]'] = f'{df.age.median():.0f} [{df.age.quantile(.25):.0f}–{df.age.quantile(.75):.0f}]'
    for c in CLASSES:
        out[f'{c}, n (%)'] = pc(int((df.dx==c).sum()), n)
    return pd.Series(out, name=name)

tabS1 = pd.concat([desc(ham,'HAM10000'), desc(isic,'ISIC 2019')] if len(isic)
                  else [desc(ham,'HAM10000')], axis=1)
print(tabS1.to_string()); tabS1.to_csv(os.path.join(RES,'table_cohort.csv'))

fig, ax = plt.subplots(1, 2 if len(isic) else 1, figsize=(11 if len(isic) else 6, 4))
axes = ax if len(isic) else [ax]
for a, df, nm, pal in zip(axes, ([ham, isic] if len(isic) else [ham]),
                          (['HAM10000','ISIC 2019'] if len(isic) else ['HAM10000']),
                          (['#2b6a99','#c0532b'])):
    v = df.dx.value_counts().reindex(CLASSES)
    a.bar(range(len(CLASSES)), v.values, color=pal); a.set_xticks(range(len(CLASSES)))
    a.set_xticklabels([c.upper() for c in CLASSES]); a.set_title(f'{nm} (n={len(df)})', loc='left', fontweight='bold')
    for i,val in enumerate(v.values): a.text(i, val+5, str(val), ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_class_distribution.png'), bbox_inches='tight'); plt.show()

# ===== Cell 7 =====
# ===== Cell 4 — Dataset, loaders, loss =====
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
T_TRAIN = transforms.Compose([transforms.RandomResizedCrop(CFG['img'], scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15), transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
T_EVAL = transforms.Compose([transforms.Resize((CFG['img'],CFG['img'])),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

class SkinDS(Dataset):
    def __init__(self, files, labels, train):
        self.files, self.labels, self.t = list(files), np.asarray(labels), (T_TRAIN if train else T_EVAL)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return self.t(Image.open(self.files[i]).convert('RGB')), int(self.labels[i])

def loaders(df, tr, va, te, bs=CFG['bs']):
    mk = lambda ix, train: SkinDS(df.file.values[ix], df.y.values[ix], train)
    return (DataLoader(mk(tr,True),  bs, shuffle=True,  num_workers=0, drop_last=True),
            DataLoader(mk(va,False), 64, num_workers=0),
            DataLoader(mk(te,False), 64, num_workers=0))

def cb_focal(logits, y, counts, beta=CFG['cb_beta'], gamma=CFG['focal_gamma']):
    # Cui et al. class-balanced focal loss.
    eff = 1.0 - np.power(beta, counts); w = (1.0 - beta) / np.maximum(eff, 1e-9)
    w = torch.tensor(w / w.sum() * len(w), dtype=torch.float32, device=logits.device)
    logp = F.log_softmax(logits, 1); p = logp.exp()
    ce = F.nll_loss(logp, y, reduction='none'); pt = p.gather(1, y.unsqueeze(1)).squeeze(1)
    return (w[y] * (1 - pt).pow(gamma) * ce).mean()
print('Pipeline ready.')

# ===== Cell 9 =====
# ===== Cell 5 — Backbones and ensemble =====
def build(name, nc=7):
    if name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights='DEFAULT'); head = m.classifier[1] = nn.Linear(1280, nc); new = m.classifier
    elif name == 'MobileNetV2':
        m = models.mobilenet_v2(weights='DEFAULT'); head = m.classifier[1] = nn.Linear(1280, nc); new = m.classifier
    elif name == 'MobileNetV3-Small':
        m = models.mobilenet_v3_small(weights='DEFAULT'); in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, nc); new = m.classifier
    elif name == 'ShuffleNetV2':
        m = models.shufflenet_v2_x1_0(weights='DEFAULT'); m.fc = nn.Linear(m.fc.in_features, nc); new = m.fc
    elif name == 'ResNet-50':
        m = models.resnet50(weights='DEFAULT'); m.fc = nn.Linear(m.fc.in_features, nc); new = m.fc
    elif name == 'DenseNet-121':
        m = models.densenet121(weights='DEFAULT'); m.classifier = nn.Linear(m.classifier.in_features, nc); new = m.classifier
    else:
        raise ValueError(name)
    return m, list(new.parameters() if isinstance(new, nn.Module) else new)

BACKBONES = ['EfficientNet-B0','MobileNetV2','MobileNetV3-Small',
             'ShuffleNetV2','ResNet-50','DenseNet-121']
POSTER3   = ['EfficientNet-B0','MobileNetV2','MobileNetV3-Small']
for n in BACKBONES:
    m,_ = build(n); print(f'{n:22s} {sum(p.numel() for p in m.parameters())/1e6:6.2f} M params')
del m

# ===== Cell 11 =====
# ===== Cell 6 — Training, metrics, bootstrap, temperature scaling =====
@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Y, Z = [], [], []
    for x,y in loader:
        with autocast('cuda', dtype=torch.float16, enabled=device.type=='cuda'):
            z = model(x.to(device, non_blocking=True))
        Z.append(z.float().cpu().numpy()); P.append(F.softmax(z.float(),1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(Z)

def fit_temperature(logits, y):
    import scipy.optimize as so
    def nll(T):
        T = max(float(T),1e-3)
        lp = logits/T - np.log(np.sum(np.exp(logits/T - logits.max(1,keepdims=True)), 1, keepdims=True)) - logits.max(1,keepdims=True)*0  # not used below
        # safer: use torch softmax
        import torch as _t
        lg = _t.tensor(logits/T, dtype=_t.float32); yy = _t.tensor(y, dtype=_t.long)
        return F.cross_entropy(lg, yy).item()
    r = so.minimize_scalar(nll, bounds=(0.05, 10), method='bounded')
    return float(r.x)

def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred==y).astype(float); e=0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1], np.linspace(0,1,bins+1)[1:]):
        m = (conf>lo) & (conf<=hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def metrics_all(P, y, nc=7):
    pred = P.argmax(1); k = min(3, nc)
    try: auc = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(nc)))
    except Exception: auc = np.nan
    return {'Accuracy': accuracy_score(y,pred), 'Balanced acc.': balanced_accuracy_score(y,pred),
            'Macro-F1': f1_score(y,pred, average='macro', labels=list(range(nc)), zero_division=0),
            'Weighted-F1': f1_score(y,pred, average='weighted', labels=list(range(nc)), zero_division=0),
            'Top-3 acc.': float(np.mean([y[i] in np.argsort(P[i])[-k:] for i in range(len(y))])),
            'Kappa': cohen_kappa_score(y,pred), 'MCC': matthews_corrcoef(y,pred),
            'AUROC': auc, 'ECE': ece(P,y)}

def boot_ix(n, B, seed=7):
    r = np.random.RandomState(seed); return [r.randint(0,n,n) for _ in range(B)]

def boot_ci(P, y, nc, B, keys=('Accuracy','Macro-F1','AUROC','Balanced acc.')):
    ix = boot_ix(len(y), B); vals = {k:[] for k in keys}
    for i in ix:
        if len(np.unique(y[i])) < 2: continue
        m = metrics_all(P[i], y[i], nc)
        for k in keys: vals[k].append(m[k])
    return {k:(np.nanpercentile(v,2.5), np.nanpercentile(v,97.5)) for k,v in vals.items()}

def train_model(name, loaders_, counts, seed, log=False):
    set_seed(seed); trl, val, _ = loaders_
    model, new = build(name); model.to(device)
    ids = {id(p) for p in new}; old = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{'params':old, 'lr':CFG['lr']},
                             {'params':new, 'lr':CFG['lr']*CFG['head_lr_mult']}], weight_decay=CFG['wd'])
    steps = CFG['epochs']*len(trl); warm = len(trl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda s: (s+1)/warm if s<warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm))))
    scaler = GradScaler(enabled=device.type=='cuda'); best, best_state, bad = -1, None, 0
    for ep in range(CFG['epochs']):
        model.train(); tl, n = 0, 0
        for x,y in trl:
            x = x.to(device, non_blocking=True); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.float16, enabled=device.type=='cuda'):
                z = model(x); loss = cb_focal(z, y, counts)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()*len(y); n += len(y)
        p, yv, _ = predict(model, val); f1 = f1_score(yv, p.argmax(1), average='macro')
        if log: print(f'  {name} ep{ep+1:02d} loss {tl/n:.3f} valF1 {f1:.3f}')
        if f1 > best: best, bad, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= CFG['patience']: break
    model.load_state_dict(best_state); return model, best

print('Engine ready.')

# ===== Cell 13 =====
# ===== Cell 7 — Run 3-fold internal CV + external evaluation =====
def run_cv(names, log=True):
    df = ham.reset_index(drop=True); y = df.y.values; groups = df.lesion_id.values
    skf = StratifiedGroupKFold(n_splits=CFG['folds'], shuffle=True, random_state=CFG['seed'])
    folds = list(skf.split(df, y, groups))
    out, temps, ext = {}, {}, {}
    for name in names:
        tag = name.replace('/','-').replace(' ','_')
        fn = os.path.join(RES, f'oof_{tag}.npz')
        if os.path.exists(fn):
            z = np.load(fn)
            out[name] = (z['P'], z['y'], z['Z'], z['FO']); temps[name] = float(z['T']); ext[name] = (z['Pe'], z['ye'])
            if log: print(f'[cached] {name}')
            continue
        P = np.zeros((len(df), 7), np.float32); Z = np.zeros((len(df), 7), np.float32); FO = np.zeros(len(df), int)
        T_sum = 0.0
        t0 = time.time()
        for k,(trv,te) in enumerate(folds):
            tr, va = train_test_split(trv, test_size=0.10, stratify=y[trv], random_state=CFG['seed']+k)
            cnt = np.bincount(y[tr], minlength=7)
            ld = loaders(df, tr, va, te)
            m, bv = train_model(name, ld, cnt, 1000+k, log=False)
            # test probs + logits
            p_te, y_te, z_te = predict(m, ld[2])
            # temperature from validation
            p_va, y_va, z_va = predict(m, ld[1]); T = fit_temperature(z_va, y_va)
            z_cal = z_te / T; p_cal = np.exp(z_cal - z_cal.max(1,keepdims=True))
            p_cal = p_cal / p_cal.sum(1,keepdims=True)
            P[te] = p_cal; Z[te] = z_te; FO[te] = k; T_sum += T
            if log: print(f'  {name} fold {k+1}/{CFG["folds"]}: valF1 {bv:.3f}  testF1 {f1_score(y_te, p_te.argmax(1), average="macro"):.3f}  T={T:.2f}', flush=True)
            del m; torch.cuda.empty_cache()
        T_mean = T_sum / CFG['folds']
        Pe = np.zeros((0,7)); ye = np.zeros(0, int)
        np.savez(fn, P=P, y=y, Z=Z, FO=FO, T=T_mean, Pe=Pe, ye=ye)
        out[name] = (P, y, Z, FO); temps[name] = T_mean; ext[name] = (Pe, ye)
        if log: print(f'== {name} OOF macroF1 {f1_score(y, P.argmax(1), average="macro"):.3f}  [{(time.time()-t0)/60:.1f} min]', flush=True)
    return out, temps, ext

RES_CV, TEMPS, RES_EXT = run_cv(BACKBONES)
# Use independent external predictions generated by the first-fold script.
for name in BACKBONES:
    ext_p = os.path.join(RES, f'ext3_independent_{name.replace(" ","_")}.npz')
    z = np.load(ext_p)
    if (not np.array_equal(z['image_id'].astype(str), isic.image_id.astype(str).values)
            or not np.array_equal(z['y'], isic.y.values)):
        raise ValueError(f'External cache does not match the independent ISIC cohort: {ext_p}')
    RES_EXT[name] = (z['P'], z['y'])

# ===== Cell 15 =====
# ===== Cell 8 — Build and evaluate ensembles =====
def soft_vote(models_probs):            # dict name -> P[N,7]
    arr = np.stack(list(models_probs.values()), 0)
    return arr.mean(0)

def weighted_vote(models_probs, weights):
    arr = np.stack([models_probs[k] for k in weights.keys()], 0)
    w = np.array([weights[k] for k in weights.keys()]); w = w/w.sum()
    return (arr * w[:,None,None]).sum(0)

# Validation-set macro-F1 per model (pooled OOF)
y = RES_CV[BACKBONES[0]][1]
val_f1 = {n: f1_score(y, RES_CV[n][0].argmax(1), average='macro') for n in BACKBONES}
tau = 10.0
w_wv = {n: math.exp(tau*val_f1[n]) for n in BACKBONES}
s = sum(w_wv.values()); w_wv = {n: v/s for n,v in w_wv.items()}
print('Learned ensemble weights:', {k: round(v,3) for k,v in w_wv.items()})

ENS = {
    'Ensemble 3 (poster, soft)':        ('soft', POSTER3),
    'Ensemble 3 (poster, weighted)':    ('wv', POSTER3),
    'Ensemble 6 (all, soft)':           ('soft', BACKBONES),
    'Ensemble 6 (all, weighted)':       ('wv', BACKBONES),
}
RES_ENS = {}
for name, (kind, members) in ENS.items():
    probs = {n: RES_CV[n][0] for n in members}
    P = soft_vote(probs) if kind=='soft' else weighted_vote(probs, {n:w_wv[n] for n in members})
    RES_ENS[name] = (P, y)
    # external
    if len(isic):
        probs_e = {n: RES_EXT[n][0] for n in members}
        Pe = soft_vote(probs_e) if kind=='soft' else weighted_vote(probs_e, {n:w_wv[n] for n in members})
        RES_ENS[name] = (P, y, Pe, isic.y.values)

print('Ensembles computed:', list(RES_ENS.keys()))

# ===== Cell 17 =====
# ===== Cell 9 — Table 1 (internal) =====
def fmt(v, ci, d=3): return f'{v:.{d}f} ({ci[0]:.{d}f}–{ci[1]:.{d}f})'

def row(name, P, y_, nc=7, B=CFG['n_boot']):
    m = metrics_all(P, y_, nc); ci = boot_ci(P, y_, nc, B)
    return {'Model': name, 'Accuracy': fmt(m['Accuracy'], ci['Accuracy']),
            'Balanced acc.': fmt(m['Balanced acc.'], ci['Balanced acc.']),
            'Macro-F1': fmt(m['Macro-F1'], ci['Macro-F1']),
            'AUROC': fmt(m['AUROC'], ci['AUROC']),
            'Top-3 acc.': f'{m["Top-3 acc."]:.3f}',
            'Kappa': f'{m["Kappa"]:.3f}', 'MCC': f'{m["MCC"]:.3f}',
            'ECE': f'{m["ECE"]:.3f}', '_f1': m['Macro-F1']}

rows = [row(n, RES_CV[n][0], RES_CV[n][1]) for n in BACKBONES]
rows += [row(n, RES_ENS[n][0], RES_ENS[n][1]) for n in RES_ENS]
T1 = pd.DataFrame(rows).drop(columns='_f1')
T1.to_csv(os.path.join(RES,'table_internal.csv'), index=False)
print('TABLE 1 – Internal lesion-grouped OOF performance'); print(T1.to_string(index=False))

# ===== Cell 19 =====
# ===== Cell 10 — Table 2 (external zero-shot) =====
if len(isic):
    rows = []
    for n in BACKBONES:
        Pe, ye = RES_EXT[n]
        if Pe.size: rows.append(row(n, Pe, ye))
    for n in RES_ENS:
        if len(RES_ENS[n]) == 4:
            _, _, Pe, ye = RES_ENS[n]; rows.append(row(n, Pe, ye))
    T2 = pd.DataFrame(rows).drop(columns='_f1')
    T2.to_csv(os.path.join(RES,'table_external.csv'), index=False)
    print('TABLE 2 – External zero-shot performance on ISIC 2019'); print(T2.to_string(index=False))
else:
    T2 = None; print('ISIC 2019 not available – Table 2 will be produced in a follow-up run.')

# ===== Cell 21 =====
# ===== Cell 11 — Paired bootstrap of ensemble vs backbones =====
def paired_boot(Pa, Pb, y_, B):
    ix = boot_ix(len(y_), B, seed=11); d = []
    for i in ix:
        if len(np.unique(y_[i]))<2: continue
        d.append(f1_score(y_[i], Pa[i].argmax(1), average='macro')
                 - f1_score(y_[i], Pb[i].argmax(1), average='macro'))
    d = np.array(d); obs = f1_score(y_, Pa.argmax(1), average='macro') - f1_score(y_, Pb.argmax(1), average='macro')
    p = 2*min((d<=0).mean(), (d>=0).mean()); return obs, np.percentile(d,2.5), np.percentile(d,97.5), max(p, 1/len(d))

def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run=0
    for r,i in enumerate(o): run = max(run, (len(ps)-r)*ps[i]); adj[i]=min(1,run)
    return adj

def pfmt(p): return '<.001' if p<.001 else (f'{p:.3f}'.lstrip('0') if p<.01 else f'{p:.2f}'.lstrip('0') if p<.99 else '>.99')

TOP = max(RES_ENS, key=lambda n: f1_score(RES_ENS[n][1], RES_ENS[n][0].argmax(1), average='macro'))
print('Reference ensemble:', TOP)
Pref, y_ = RES_ENS[TOP][0], RES_ENS[TOP][1]
rows = []
for n in BACKBONES:
    d, lo, hi, p = paired_boot(Pref, RES_CV[n][0], y_, CFG['n_boot'])
    rows.append([n, d, lo, hi, p])
adj = holm([r[4] for r in rows])
T3 = pd.DataFrame([{'Backbone': r[0], 'ΔMacro-F1': f'{r[1]:+.3f} ({r[2]:+.3f} to {r[3]:+.3f})',
                    'P (raw)': pfmt(r[4]), 'P (Holm)': pfmt(a)} for r,a in zip(rows,adj)])
T3.to_csv(os.path.join(RES,'table_paired.csv'), index=False); print(T3.to_string(index=False))

# ===== Cell 23 =====
# ===== Cell 12 — Ablation: every pair + every triple of the 3 poster backbones + each single + all 6 =====
from itertools import combinations
abl = []
for r in (1,2,3):
    for combo in combinations(POSTER3, r):
        P = soft_vote({n: RES_CV[n][0] for n in combo})
        abl.append(row(' + '.join(combo) + ' (soft)', P, y_))
P_all = soft_vote({n: RES_CV[n][0] for n in BACKBONES})
abl.append(row('All 6 (soft, ours)', P_all, y_))
T4 = pd.DataFrame(abl).drop(columns='_f1')
T4.to_csv(os.path.join(RES,'table_ablation.csv'), index=False)
print('TABLE 4 – Ensemble ablation'); print(T4.to_string(index=False))

# ===== Cell 25 =====
# ===== Cell 13 — Per-class performance + calibration =====
P_top = RES_ENS[TOP][0]
pr, rc, f1, sup = precision_recall_fscore_support(y_, P_top.argmax(1), labels=range(7), zero_division=0)
T5 = pd.DataFrame({'Class':[c.upper() for c in CLASSES], 'n':sup,
                   'Precision':pr.round(3), 'Recall':rc.round(3), 'F1':f1.round(3)})
T5.to_csv(os.path.join(RES,'table_perclass.csv'), index=False); print('TABLE 5 – Per-class'); print(T5.to_string(index=False))

cal_rows = []
for n in BACKBONES:
    P_raw_cv = []  # raw (no temperature) probs for comparison
    df_cv = ham; y_all = df_cv.y.values
    # recompute raw probs from stored logits
    P_raw = F.softmax(torch.tensor(RES_CV[n][2]), dim=1).numpy()
    cal_rows.append({'Model': n, 'ECE raw': f'{ece(P_raw, y_all):.3f}',
                     'T': f'{TEMPS[n]:.2f}', 'ECE cal': f'{ece(RES_CV[n][0], y_all):.3f}'})
T6 = pd.DataFrame(cal_rows)
T6.to_csv(os.path.join(RES,'table_calibration.csv'), index=False); print('TABLE 6 – Calibration'); print(T6.to_string(index=False))

# ===== Cell 27 =====
# ===== Cell 14 — Params, GMACs, latency =====
from torch.utils.flop_counter import FlopCounterMode
def profile(name):
    import gc
    gc.collect(); torch.cuda.empty_cache()
    m,_ = build(name); m.eval()
    params = sum(p.numel() for p in m.parameters())/1e6
    x = torch.randn(1,3,224,224)
    with torch.no_grad(), FlopCounterMode(display=False) as fc: m(x)
    gmacs = fc.get_total_flops()/2e9
    lat = {}
    for dev in (['cpu'] + (['cuda'] if torch.cuda.is_available() else [])):
        mm = m.to(dev); xx = x.to(dev)
        with torch.no_grad():
            for _ in range(5): mm(xx)
            if dev=='cuda': torch.cuda.synchronize()
            t0 = time.time(); R = 20 if dev=='cpu' else 100
            for _ in range(R): mm(xx)
            if dev=='cuda': torch.cuda.synchronize()
        lat[dev] = (time.time()-t0)/R*1000
    del m, mm; gc.collect(); torch.cuda.empty_cache()
    return params, gmacs, lat

rows = []
for n in BACKBONES:
    p, g, l = profile(n)
    rows.append({'Model': n, 'Params (M)': round(p,2), 'GMACs': round(g,2),
                 'CPU ms': round(l['cpu'],1), 'GPU ms': round(l.get('cuda',float('nan')),2)})
# Ensemble = sum of its members
for ename, (kind, members) in ENS.items():
    totp = sum(dict(zip(['Params (M)' for _ in rows], [r['Params (M)'] for r in rows if r['Model'] in members])).values())
    # recompute cleanly
    totp = sum(r['Params (M)'] for r in rows if r['Model'] in members)
    totg = sum(r['GMACs']       for r in rows if r['Model'] in members)
    totcpu = sum(r['CPU ms']    for r in rows if r['Model'] in members)
    totgpu = sum(r['GPU ms']    for r in rows if r['Model'] in members)
    rows.append({'Model': ename, 'Params (M)': round(totp,2), 'GMACs': round(totg,2),
                 'CPU ms': round(totcpu,1), 'GPU ms': round(totgpu,2)})
T7 = pd.DataFrame(rows); T7.to_csv(os.path.join(RES,'table_compute.csv'), index=False)
print('TABLE 7 – Compute'); print(T7.to_string(index=False))

# ===== Cell 29 =====
# ===== Cell 15 — Figure 3 (forest), Figure 4 (confusion), Figure 5 (ROC), Figure 6 (calibration), Figure 7 (per-class delta) =====
from sklearn.metrics import roc_curve
all_results = {n:(RES_CV[n][0], RES_CV[n][1]) for n in BACKBONES} | {n:(RES_ENS[n][0], RES_ENS[n][1]) for n in RES_ENS}

# Figure 3 – forest plot
def forest(ax, results, title):
    data = []
    for n,(P,yy) in results.items():
        m = metrics_all(P, yy); ci = boot_ci(P, yy, 7, CFG['n_boot'], keys=('Macro-F1',))
        data.append((n, m['Macro-F1'], *ci['Macro-F1']))
    data.sort(key=lambda t:t[1])
    for i,(n,v,lo,hi) in enumerate(data):
        is_ens = 'Ensemble' in n
        ax.errorbar(v, i, xerr=[[v-lo],[hi-v]], fmt='o',
                    color='#c0392b' if is_ens else '#34495e',
                    ms=8 if is_ens else 6, capsize=3, lw=2 if is_ens else 1.4)
    ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data])
    ax.set_xlabel('Macro-F1 (95% CI)'); ax.set_title(title, loc='left', fontweight='bold')
    ax.grid(axis='x', ls=':', alpha=.6)

fig, ax = plt.subplots(1, 2 if len(isic) else 1, figsize=(14 if len(isic) else 7, 5.2))
axes = ax if len(isic) else [ax]
forest(axes[0], all_results, '(a) Internal (HAM10000)')
if len(isic):
    ext_results = {n:(RES_EXT[n][0], RES_EXT[n][1]) for n in BACKBONES} | {n:(RES_ENS[n][2], RES_ENS[n][3]) for n in RES_ENS if len(RES_ENS[n])==4}
    forest(axes[1], ext_results, '(b) External zero-shot (ISIC 2019)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_forest.png'), bbox_inches='tight'); plt.show()

# Figure 4 – confusion matrix (ensemble)
P, yy = RES_ENS[TOP][0], RES_ENS[TOP][1]; cm = confusion_matrix(yy, P.argmax(1))
cmn = cm / np.maximum(cm.sum(1,keepdims=True), 1)
plt.figure(figsize=(6.6,5.6))
sns.heatmap(cmn, annot=np.array([[f'{cmn[i,j]*100:.0f}%\n(n={cm[i,j]})' for j in range(7)] for i in range(7)]),
            fmt='', cmap='Blues', xticklabels=[c.upper() for c in CLASSES], yticklabels=[c.upper() for c in CLASSES],
            cbar_kws={'label':'Row-normalised'})
plt.xlabel('Predicted'); plt.ylabel('Reference'); plt.title(TOP, loc='left', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_confusion.png'), bbox_inches='tight'); plt.show()

# Figure 5 – ROC (one-vs-rest) + melanoma zoom
from sklearn.preprocessing import label_binarize
Y = label_binarize(yy, classes=list(range(7)))
fig, ax = plt.subplots(1, 2, figsize=(11,4.6))
colors = sns.color_palette('tab10', 7)
for k,c in enumerate(CLASSES):
    fpr,tpr,_ = roc_curve(Y[:,k], P[:,k])
    auc = roc_auc_score(Y[:,k], P[:,k])
    ax[0].plot(fpr, tpr, color=colors[k], label=f'{c.upper()} AUC={auc:.2f}')
ax[0].plot([0,1],[0,1],'k:'); ax[0].set_title('(a) One-vs-rest ROC', loc='left', fontweight='bold')
ax[0].set_xlabel('1 – specificity'); ax[0].set_ylabel('Sensitivity'); ax[0].legend(frameon=False, fontsize=8)
# Melanoma binary
mel_idx = CLASSES.index('mel'); y_mel = (yy==mel_idx).astype(int); p_mel = P[:,mel_idx]
fpr,tpr,_ = roc_curve(y_mel, p_mel); auc = roc_auc_score(y_mel, p_mel)
ax[1].plot(fpr, tpr, color='#c0392b', lw=2, label=f'Melanoma (AUROC={auc:.3f})')
ax[1].plot([0,1],[0,1],'k:'); ax[1].set_title('(b) Melanoma detection', loc='left', fontweight='bold')
ax[1].set_xlabel('1 – specificity'); ax[1].set_ylabel('Sensitivity'); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_roc.png'), bbox_inches='tight'); plt.show()

# Figure 6 – reliability
fig, ax = plt.subplots(figsize=(6,4.5))
for n, col in zip([TOP, BACKBONES[0]], ['#c0392b','#34495e']):
    Pp, yp = all_results[n]
    conf, acc = Pp.max(1), (Pp.argmax(1)==yp); bins = np.linspace(0,1,11); xs, ys = [], []
    for lo,hi in zip(bins[:-1], bins[1:]):
        m = (conf>lo) & (conf<=hi)
        if m.sum()>=5: xs.append(conf[m].mean()); ys.append(acc[m].mean())
    ax.plot(xs, ys, 'o-', color=col, label=f'{n} (ECE {ece(Pp, yp):.3f})')
ax.plot([0,1],[0,1],'k:'); ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
ax.legend(frameon=False); ax.set_title('Reliability diagram', loc='left', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_calibration.png'), bbox_inches='tight'); plt.show()

# ===== Cell 31 =====
# ===== Cell 16 — Grad-CAM on EfficientNet-B0 trained on fold 1 =====
def gradcam(model, x, cls, layer):
    model.eval(); store = {}
    def fwd(m,i,o): store['a'] = o; o.register_hook(lambda g: store.__setitem__('g', g))
    h = layer.register_forward_hook(fwd)
    out = model(x); model.zero_grad(); out[0, cls].backward(); h.remove()
    w = store['g'].mean((2,3), keepdim=True); cam = F.relu((w*store['a']).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0,0]
    cam = cam - cam.min(); return (cam / (cam.max()+1e-8)).detach().cpu().numpy()

df = ham.reset_index(drop=True); y = df.y.values; groups = df.lesion_id.values
skf = StratifiedGroupKFold(n_splits=CFG['folds'], shuffle=True, random_state=CFG['seed'])
folds = list(skf.split(df, y, groups)); trv, te = folds[0]
tr, va = train_test_split(trv, test_size=0.10, stratify=y[trv], random_state=CFG['seed'])
cnt = np.bincount(y[tr], minlength=7); ld = loaders(df, tr, va, te)
gc_model, _ = train_model('EfficientNet-B0', ld, cnt, 1000, log=False)

fig, axs = plt.subplots(2, 7, figsize=(14, 4.4))
layer = gc_model.features[-1]
for j in range(7):
    cand_ix = [i for i in range(len(ld[2].dataset)) if ld[2].dataset.labels[i]==j]
    if not cand_ix: axs[0,j].axis('off'); axs[1,j].axis('off'); continue
    i = cand_ix[0]
    x_, _ = ld[2].dataset[i]; xb = x_.unsqueeze(0).to(device).requires_grad_(True)
    cam = gradcam(gc_model, xb, j, layer)
    im = np.clip(x_.numpy().transpose(1,2,0)*np.array(STD)+np.array(MEAN), 0, 1)
    axs[0,j].imshow(im); axs[0,j].axis('off'); axs[0,j].set_title(CLASSES[j].upper(), fontsize=9)
    axs[1,j].imshow(im); axs[1,j].imshow(cam, cmap='jet', alpha=.45); axs[1,j].axis('off')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_gradcam.png'), bbox_inches='tight'); plt.show()
del gc_model; torch.cuda.empty_cache()

# ===== Cell 33 =====
# ===== Cell 17 — summary.json =====
SUMMARY = dict(config=CFG, n_ham=int(len(ham)), n_isic=int(len(isic)),
               classes=CLASSES, reference_ensemble=TOP,
               backbones={n:{'ValMacroF1': float(val_f1[n]), 'EnsembleWeight': float(w_wv[n]),
                             'Temperature': float(TEMPS[n])} for n in BACKBONES},
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
json.dump(SUMMARY, open(os.path.join(RES,'summary.json'),'w'), indent=2, default=float)
print(json.dumps(SUMMARY, indent=1, default=float))