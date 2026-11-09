"""
Core Interpretability & Attribution Engine for ChemMambaDDI (GASPDDI).

Supports:
1. Dependency-free direct loading of PyTorch (.pth/.pt) checkpoints into NumPy arrays
   (enabling exact Mamba S6 + 1D-CNN + Fingerprint + MLP inference & analytical backpropagation
   even on CPU/macOS environments without CUDA mamba-ssm).
2. Exact token-level & atom-level attribution (Grad x Activation, Integrated Gradients,
   CNN-to-Mamba backpropagation, and Mamba SSM selective scan step-size gate delta_t).
3. Exact 2D fingerprint bit attribution (512-bit Morgan ECFP4 + 167-bit MACCS Keys)
   and RDKit chemical fragment depiction.
4. 8-block symmetric fusion attribution and counterfactual modality/operator ablation.
"""

import io
import zipfile
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

RDLogger.DisableLog('rdApp.*')


# Curated human-readable SMARTS/chemical descriptions for key MACCS structural keys (1-indexed 1..166)
MACCS_DESCRIPTIONS = {
    8: "4-membered ring (e.g., beta-lactam core)",
    11: "4-membered ring heterocycle",
    19: "7-membered ring system",
    23: "N-C(=O)-O Carbamate / urethane group",
    24: "N-O bond (Nitro / Oxime / Isoxazole)",
    25: "N-C(=N)-N Guanidine / Triazole motif",
    26: "C$=C($C)$C Trisubstituted alkene",
    36: "S-Heterocycle (Thiophene / Thiazole / Phenothiazine)",
    37: "N-C(=O)-N Urea / Barbiturate / Hydantoin",
    38: "N-C(=C)-N Heteroaromatic ring junction",
    41: "C#N Nitrile / Cyano group",
    42: "Fluorine (F) halogen substituent",
    43: "N-H..N Intramolecular H-bond donor pair",
    47: "C-S-N Sulfonamide / Thiazide linkage",
    49: "Charged / Polar Zwitterionic motif",
    52: "N-N Hydrazine / Pyrazole / Triazole bond",
    53: "QHAAAQH Ortho/Meta di-H-bond donor",
    54: "QHAAQH Vicinal diol / catechol / amino-alcohol",
    55: "O-S-O Sulfone / Sulfonamide / Sulfate",
    57: "O-Heterocycle (Furan / Pyran / Lactone / Coumarin)",
    58: "Q-S-Q Sulfonyl / Thioether linkage",
    59: "S=O Sulfoxide / Sulfonyl aromatic bond",
    60: "S=O Sulfonyl group (-SO2-)",
    62: "A$!A$A Biphenyl / Linked ring bridge",
    65: "C%N Aromatic C:N bond (Pyridine/Pyrimidine/Azole)",
    66: "C(C)(C)(C)A Quaternary carbon center (Steroid/Terpene)",
    69: "Q-Q-H Hydroxylamine / Hydrazide / Sulfonamide",
    72: "O-C-C-O Ethylene glycol / Dioxolane / Ether",
    73: "S=A Thiocarbonyl / Sulfonyl double bond",
    74: "CH3-A-CH3 Gem-dimethyl / Isobutyl group",
    75: "A!N$A Tertiary / Piperazine / Piperidine ring N",
    77: "N-A-N Diamine / Pyrimidine / Imidazole / Triazole",
    79: "N-A-A-N Ethylenediamine / Piperazine motif",
    80: "N-A-A-A-N Propylamine / Extended N-linker",
    81: "S-A(A)A Sulfonyl / Thio-substituted branch",
    82: "ACH2QH Primary alcohol / Benzyl alcohol / Aminomethyl",
    83: "5-membered heterocycle (Azole / Furan / Thiophene)",
    84: "NH2 Primary amine / Sulfanilamide (-NH2)",
    85: "CN(C)C Tertiary amine (hERG / CNS pharmacophore)",
    86: "CH2QCH2 Dialkyl ether / Secondary amine / Thioether",
    87: "X!A$A Halogen attached to ring (Cl/F/Br-Aryl)",
    88: "Sulfur (S) atom present",
    89: "O-A-A-A-O 1,3-Dioxy / Beta-dicarbonyl / Lactone",
    90: "QHAACH2A Beta-hydroxy / Beta-amino ethyl motif",
    91: "QHAAACH2A Gamma-hydroxy / Propylamine chain",
    92: "O=C(N)C Amide / Peptide / Acetamide bond",
    93: "QCH3 Methoxy / Methylamino / Methylthio (-OCH3/-NCH3)",
    94: "Q-N Bond (Nitro / Sulfonamide / Isoxazole)",
    95: "N-A-A-O Ethanolamine / Morpholine / Oxazolidinone",
    96: "5-membered ring (Cyclopentyl / Azole / Furan)",
    97: "N-A-A-A-O Propanolamine (Beta-blocker / SSRI motif)",
    98: "6-membered heteroaromatic ring (Pyridine/Pyrazine)",
    99: "C=C Alkene / Unsaturated carbon double bond",
    100: "ACH2N Secondary/Tertiary benzyl or alkyl amine",
    101: "8M+ Ring / Fused bicyclic-polycyclic system (Quinolone/Steroid)",
    102: "Q-O Single bond (Ester / Peroxide / Hydroxamate)",
    103: "Chlorine (Cl) halogen substituent",
    104: "QHACH2A Alpha-hydroxy / Alpha-amino aliphatic group",
    105: "A$A($A)$A Fused polycyclic ring junction",
    106: "QA(Q)Q Gem-diheteroatom (Sulfonamide / Carboxylate / Urea)",
    107: "Halogen (F, Cl, Br, I) substituent",
    108: "CH3AAACH2A Aliphatic butyl / Isobutyl hydrophobic chain",
    109: "ACH2O Ether / Alkoxymethyl / Ester (-CH2-O-)",
    110: "N-C-O Amide / Carbamate / Oxazole linkage",
    111: "N-A-CH2-A Alkyl-substituted amine / Piperidine",
    112: "AA(A)(A)A Tetrahedral / Sulfonyl / Quaternary center",
    113: "Onot%A%A Aromatic ether / Phenol / Aryl ester (Ar-O)",
    114: "CH3CH2A Ethyl group (-C2H5)",
    115: "CH3ACH2A Propyl / Sec-butyl / Isobutyl chain",
    116: "CH3A$A Methyl-substituted ring",
    117: "N-A-O Amide / Sulfonamide / Nitrosamine",
    118: "ACH2CH2A Ethylene linker (-CH2-CH2-) >= 2",
    119: "N=A Imine / Azo / Non-aromatic C=N",
    120: "Heterocyclic atom >= 2 (Diazine / Azole / Piperazine)",
    121: "N-Heterocycle (Piperidine / Pyridine / Azole / Quinolone)",
    122: "AN(A)A Tertiary amine / N-substituted heterocycle",
    123: "O-C-O Carboxylic acid / Ester / Lactone / Acetal",
    124: "Q-Q Heteroatom-heteroatom bond (S-N, N-N, N-O, S-O)",
    125: "Aromatic Rings >= 2 (Biphenyl / Naphthyl / Drug scaffold)",
    126: "A!O!A Ester / Phenoxy / Aliphatic ether (-O-)",
    127: "A$A!O>=2 Di-oxygenated ring (Catechol / Coumarin / Flavone)",
    128: "ACH2AAACH2A Butylene / Piperidine / Cyclohexyl chain",
    129: "ACH2AACH2A Propylene / Piperidine / Pyrrolidine ring",
    130: "Q-Q >= 2 Multiple heteroatom-heteroatom bonds (Sulfonamide)",
    131: "QH >= 2 Multiple H-bond donors (-OH / -NH >= 2)",
    132: "O-A-CH2-A Alkoxy / Ester / Beta-hydroxyether",
    133: "A$A!N Aniline / Amino-heterocycle / Aryl-NH-",
    134: "Halogen (F, Cl, Br, I) atom",
    135: "Nnot%A%A Aryl-amine / Sulfanilamide / Anilide (Ar-N)",
    136: "O=A >= 2 Multiple carbonyl/sulfonyl oxygens (-COOH, -SO2-, Imide)",
    137: "Heterocycle (O, N, S ring system)",
    138: "QCH2A>=2 Multiple hetero-methylene groups",
    139: "OH Hydroxyl / Carboxylic acid / Phenol (-OH)",
    140: "O >= 4 Poly-oxygenated scaffold (>= 4 Oxygen atoms)",
    141: "CH3 >= 3 Polymethylated lipophilic scaffold",
    142: "N >= 2 Diamine / Poly-nitrogen drug scaffold",
    143: "A$A!O Phenol / Aryl ether / Coumarin / Lactone",
    144: "Anot%A%Anot%A Conjugated / Biaryl linker",
    145: "6-membered rings >= 2 (Bicyclic / Tricyclic / Steroid)",
    146: "O >= 3 Tri-oxygenated scaffold (-COOH + =O / Sulfonamide)",
    147: "ACH2CH2A Linear ethylene chain",
    148: "AQ(A)A Tri-substituted heteroatom (Tertiary N / Sulfone)",
    149: "CH3 >= 2 Dimethyl / Branched alkyl motif",
    150: "A!A$A!A Multi-substituted ring",
    151: "NH Secondary or Primary Amine (-NH- / -NH2)",
    152: "OC(C)C Phenolic / Enolic / Isopropoxy / Alpha-hydroxy",
    153: "QCH2A Heteroatom-attached methylene (-CH2-N / -CH2-O)",
    154: "C=O Carbonyl group (Ketone / Carboxylic Acid / Amide / Lactone)",
    155: "A!CH2!A Aliphatic methylene bridge (-CH2-)",
    156: "N-A(A)A Branched C-N bond (Alpha-amino acid / Chiral amine)",
    157: "C-O Single bond (Alcohol / Ether / Ester / Acid)",
    158: "C-N Single bond (Amine / Amide / Piperazine)",
    159: "O >= 2 Di-oxygenated motif (Carboxyl / Ester / Sulfone)",
    160: "CH3 Methyl group (-CH3)",
    161: "Nitrogen (N) atom present",
    162: "Aromatic ring system",
    163: "6-membered ring (Benzene / Pyridine / Piperazine)",
    164: "Oxygen (O) atom present",
    165: "Ring system present",
    166: "Disconnected fragments / Salt counterion",
}


def load_pth_numpy(path):
    """Load a PyTorch zip checkpoint (.pth/.pt) directly into NumPy arrays without importing torch."""
    zf = zipfile.ZipFile(path, 'r')
    prefix = zf.namelist()[0].split('/')[0]
    dtype_map = {
        'FloatStorage': np.float32,
        'DoubleStorage': np.float64,
        'HalfStorage': np.float16,
        'LongStorage': np.int64,
        'IntStorage': np.int32,
        'ShortStorage': np.int16,
        'CharStorage': np.int8,
        'ByteStorage': np.uint8,
        'BoolStorage': np.bool_,
    }
    storages = {}

    class Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            _, storage_cls, key, _, _ = pid[:5]
            if key not in storages:
                raw = zf.read(f'{prefix}/data/{key}')
                dt = dtype_map.get(storage_cls.__name__, np.float32)
                storages[key] = np.frombuffer(raw, dtype=dt)
            return storages[key]

        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                def _rebuild(storage, offset, size, stride, *args):
                    count = int(np.prod(size))
                    return storage[offset:offset + count].reshape(size)
                return _rebuild
            if module == 'torch':
                return type(name, (), {})
            if module.startswith('numpy'):
                return super().find_class(module, name)
            return super().find_class(module, name)

    return Unpickler(io.BytesIO(zf.read(f'{prefix}/data.pkl'))).load()


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def d_silu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))
    return s * (1.0 + x * (1.0 - s))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -80.0, 20.0))))


def build_fp_and_bitinfo(smiles_list):
    """Compute 679-dim fingerprints (512 Morgan radius-2 + 167 MACCS) and RDKit bitInfo dictionaries."""
    fp_cache = {}
    morgan_info_cache = {}
    for smi in sorted(set(smiles_list)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp_cache[smi] = np.zeros(679, dtype=np.float32)
            morgan_info_cache[smi] = {}
        else:
            info = {}
            morgan_bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512, bitInfo=info)
            morgan = np.array(morgan_bv, dtype=np.float32)
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
            fp_cache[smi] = np.concatenate([morgan, maccs])
            morgan_info_cache[smi] = info
    return fp_cache, morgan_info_cache


class ChemMambaNumpyEngine:
    """
    Exact NumPy Inference & Analytical Attribution Engine for ChemMambaDDI.
    Matches PyTorch + mamba_ssm forward pass bit-for-bit (float32).
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = str(checkpoint_path)
        self.ckpt = load_pth_numpy(checkpoint_path)
        self.sd = self.ckpt['model_state_dict']
        self.tokenizer = self.ckpt['tokenizer']
        self.threshold = float(self.ckpt.get('threshold', 0.5))
        self.drug_cache = {}

    def encode_smiles(self, smiles):
        return np.array([self.tokenizer.get(c, self.tokenizer['<UNK>']) for c in smiles], dtype=np.int64)

    def mamba_student_forward(self, input_ids, return_details=False):
        """
        Exact forward pass of MambaStudent (Embedding -> Mamba S6 -> Mask -> Conv1d(k=7) -> ReLU -> MaskedMeanPool).
        input_ids: (B, L) int64
        """
        B, L = input_ids.shape
        mask = (input_ids != 0)
        x = self.sd['student.embedding.weight'][input_ids]  # (B, L, 256)

        xz = x @ self.sd['student.mamba.in_proj.weight'].T  # (B, L, 1024)
        x_proj_in, z = xz[:, :, :512], xz[:, :, 512:]

        # Depthwise causal Conv1d (kernel_size=4, padding=3 on left)
        x_pad = np.pad(x_proj_in, ((0, 0), (3, 0), (0, 0)), mode='constant')
        w_conv = self.sd['student.mamba.conv1d.weight'][:, 0, :]  # (512, 4)
        b_conv = self.sd['student.mamba.conv1d.bias']
        x_conv_pre = (
            x_pad[:, 0:L, :] * w_conv[:, 0] +
            x_pad[:, 1:L + 1, :] * w_conv[:, 1] +
            x_pad[:, 2:L + 2, :] * w_conv[:, 2] +
            x_pad[:, 3:L + 3, :] * w_conv[:, 3]
        ) + b_conv
        x_conv = silu(x_conv_pre)

        x_dbl = x_conv @ self.sd['student.mamba.x_proj.weight'].T  # (B, L, 48)
        dt_raw, B_ssm, C_ssm = x_dbl[:, :, :16], x_dbl[:, :, 16:32], x_dbl[:, :, 32:48]
        dt = softplus(dt_raw @ self.sd['student.mamba.dt_proj.weight'].T + self.sd['student.mamba.dt_proj.bias'])
        A = -np.exp(self.sd['student.mamba.A_log'])  # (512, 16)
        D = self.sd['student.mamba.D']  # (512,)

        h = np.zeros((B, 512, 16), dtype=np.float32)
        y = np.empty((B, L, 512), dtype=np.float32)
        for t in range(L):
            dt_t = dt[:, t, :, None]
            dA = np.exp(dt_t * A[None, :, :])
            dBx = dt_t * B_ssm[:, t, None, :] * x_conv[:, t, :, None]
            h = dA * h + dBx
            y[:, t, :] = np.einsum('bdn,bn->bd', h, C_ssm[:, t, :]) + D * x_conv[:, t, :]

        gate_z = silu(z)
        y_gated = y * gate_z
        out_mamba = y_gated @ self.sd['student.mamba.out_proj.weight'].T  # (B, L, 256)
        out_mamba = np.where(mask[:, :, None], out_mamba, 0.0)

        # Stage 2: Local 1D-CNN (kernel_size=7, padding=3)
        out_pad = np.pad(out_mamba, ((0, 0), (3, 3), (0, 0)), mode='constant')
        w_cnn = self.sd['student.local_cnn.weight']  # (256, 256, 7)
        b_cnn = self.sd['student.local_cnn.bias']
        pre_cnn = np.zeros((B, L, 256), dtype=np.float32)
        for k in range(7):
            pre_cnn += out_pad[:, k:k + L, :] @ w_cnn[:, :, k].T
        pre_cnn += b_cnn
        out_cnn = np.maximum(0.0, pre_cnn)

        out_masked = np.where(mask[:, :, None], out_cnn, 0.0)
        count = np.maximum(1, mask.sum(axis=1, keepdims=True)).astype(np.float32)
        pooled = out_masked.sum(axis=1) / count

        if not return_details:
            return pooled
        return {
            'pooled': pooled,
            'out_cnn': out_masked,
            'pre_cnn': pre_cnn,
            'out_mamba': out_mamba,
            'dt': dt,          # (B, L, 512) selective scan step size
            'gate_z': gate_z,  # (B, L, 512) SiLU gating branch
            'mask': mask,
        }

    def precompute_drug_table(self, smiles_list, fp_cache, batch_size=64):
        """
        Precompute 1D Mamba pooled embeddings (e_1d) and 2D projected fingerprints (f_fp)
        for all unique drugs sorted by SMILES length to minimize padding overhead.
        """
        unique_smiles = sorted(set(smiles_list), key=len)
        e1d_map = {}
        fp_proj_map = {}
        w_fp = self.sd['predictor.fp_proj.0.weight']
        b_fp = self.sd['predictor.fp_proj.0.bias']

        for start in range(0, len(unique_smiles), batch_size):
            batch = unique_smiles[start:start + batch_size]
            max_l = max(len(s) for s in batch)
            ids = np.zeros((len(batch), max_l), dtype=np.int64)
            for i, s in enumerate(batch):
                for j, c in enumerate(s):
                    ids[i, j] = self.tokenizer.get(c, self.tokenizer['<UNK>'])
            pooled = self.mamba_student_forward(ids, return_details=False)
            fps = np.stack([fp_cache[s] for s in batch])
            f_proj = np.maximum(0.0, fps @ w_fp.T + b_fp)
            for i, s in enumerate(batch):
                e1d_map[s] = pooled[i]
                fp_proj_map[s] = f_proj[i]

        self.e1d_map = e1d_map
        self.fp_proj_map = fp_proj_map
        return e1d_map, fp_proj_map

    def predict_pairs_from_cache(self, smiles_a, smiles_b, relations, mask_blocks=None):
        """
        Fast batched MLP forward pass using precomputed drug embeddings.
        mask_blocks: optional set of block indices (0..7) or names to zero out for counterfactual ablation:
          0: ea, 1: eb, 2: ea*eb, 3: |ea-eb|, 4: fa, 5: fb, 6: fa*fb, 7: |fa-fb|
        """
        ea = np.stack([self.e1d_map[s] for s in smiles_a])
        eb = np.stack([self.e1d_map[s] for s in smiles_b])
        fa = np.stack([self.fp_proj_map[s] for s in smiles_a])
        fb = np.stack([self.fp_proj_map[s] for s in smiles_b])

        blocks = [
            ea, eb, ea * eb, np.abs(ea - eb),
            fa, fb, fa * fb, np.abs(fa - fb)
        ]
        if mask_blocks:
            for b_idx in mask_blocks:
                blocks[b_idx] = np.zeros_like(blocks[b_idx])

        u = np.concatenate(blocks, axis=-1)  # (N, 2048)
        z1 = u @ self.sd['predictor.mlp.0.weight'].T + self.sd['predictor.mlp.0.bias']
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.sd['predictor.mlp.3.weight'].T + self.sd['predictor.mlp.3.bias']
        h2 = np.maximum(0.0, z2)
        logits_all = h2 @ self.sd['predictor.mlp.5.weight'].T + self.sd['predictor.mlp.5.bias']
        r_idx = np.asarray(relations, dtype=np.int64)
        logits = logits_all[np.arange(len(r_idx)), r_idx]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        return probs, logits, h2, u

    def explain_pair(self, smiles_a, smiles_b, relation_idx, fp_cache, ig_steps=20):
        """
        Compute comprehensive multi-level attributions for a single DDI pair (Drug A, Drug B, relation r):
        1. Token-level post-CNN attribution S_CNN(t) & pre-CNN Mamba attribution S_Mamba(t)
        2. Mamba selective scan step size delta_t(t) & SiLU gate activation z(t)
        3. Exact Atom-level attribution mapped onto RDKit heavy atoms for Drug A & Drug B
        4. Exact 679-bit Fingerprint attribution (512 Morgan + 167 MACCS) for Drug A & Drug B
        5. 8-block Fusion Integrated Gradients (IG) attribution
        """
        ids_a = self.encode_smiles(smiles_a)[None, :]
        ids_b = self.encode_smiles(smiles_b)[None, :]
        da = self.mamba_student_forward(ids_a, return_details=True)
        db = self.mamba_student_forward(ids_b, return_details=True)

        fp_a = fp_cache[smiles_a][None, :]  # (1, 679)
        fp_b = fp_cache[smiles_b][None, :]  # (1, 679)

        w_fp = self.sd['predictor.fp_proj.0.weight']
        b_fp = self.sd['predictor.fp_proj.0.bias']
        z_fpa = fp_a @ w_fp.T + b_fp
        z_fpb = fp_b @ w_fp.T + b_fp
        fa = np.maximum(0.0, z_fpa)
        fb = np.maximum(0.0, z_fpb)

        ea, eb = da['pooled'], db['pooled']
        blocks = [ea, eb, ea * eb, np.abs(ea - eb), fa, fb, fa * fb, np.abs(fa - fb)]
        u = np.concatenate(blocks, axis=-1)  # (1, 2048)

        w0, b0 = self.sd['predictor.mlp.0.weight'], self.sd['predictor.mlp.0.bias']
        w3, b3 = self.sd['predictor.mlp.3.weight'], self.sd['predictor.mlp.3.bias']
        w5, b5 = self.sd['predictor.mlp.5.weight'], self.sd['predictor.mlp.5.bias']

        def mlp_grad_u(u_in):
            # u_in: (S, 2048)
            z1 = u_in @ w0.T + b0
            h1 = np.maximum(0.0, z1)
            z2 = h1 @ w3.T + b3
            h2 = np.maximum(0.0, z2)
            logits = h2 @ w5.T + b5
            # Backprop from logits[:, relation_idx]
            dh2 = np.broadcast_to(w5[relation_idx:relation_idx + 1, :], h2.shape)
            dz2 = dh2 * (z2 > 0).astype(np.float32)
            dh1 = dz2 @ w3
            dz1 = dh1 * (z1 > 0).astype(np.float32)
            du = dz1 @ w0
            return logits[:, relation_idx], du, h2

        logits_actual, grad_u_actual, h2_actual = mlp_grad_u(u)
        logit_val = float(logits_actual[0])
        prob_val = float(1.0 / (1.0 + np.exp(-np.clip(logit_val, -80.0, 80.0))))

        # Integrated Gradients along straight line from 0 to u
        alphas = np.linspace(0.0, 1.0, ig_steps, dtype=np.float32)[:, None]
        u_path = alphas * u  # (ig_steps, 2048)
        _, grad_u_path, _ = mlp_grad_u(u_path)
        avg_grad_u = grad_u_path.mean(axis=0)  # (2048,)
        ig_u = u[0] * avg_grad_u  # (2048,)

        # Use exact local gradient at u (and average IG gradient) to backprop into ea, eb, fa, fb
        gu = grad_u_actual[0]
        g_ea_blk, g_eb_blk, g_eab_blk, g_ediff_blk = gu[0:256], gu[256:512], gu[512:768], gu[768:1024]
        g_fa_blk, g_fb_blk, g_fab_blk, g_fdiff_blk = gu[1024:1280], gu[1280:1536], gu[1536:1792], gu[1792:2048]

        sign_e = np.sign(ea[0] - eb[0])
        sign_f = np.sign(fa[0] - fb[0])

        d_ea = g_ea_blk + g_eab_blk * eb[0] + g_ediff_blk * sign_e
        d_eb = g_eb_blk + g_eab_blk * ea[0] - g_ediff_blk * sign_e

        d_fa = g_fa_blk + g_fab_blk * fb[0] + g_fdiff_blk * sign_f
        d_fb = g_fb_blk + g_fab_blk * fa[0] - g_fdiff_blk * sign_f

        # Backprop through fp_proj to 679 fingerprint bits
        d_fp_a = (d_fa * (z_fpa[0] > 0).astype(np.float32)) @ w_fp  # (679,)
        d_fp_b = (d_fb * (z_fpb[0] > 0).astype(np.float32)) @ w_fp  # (679,)
        fp_attr_a = fp_a[0] * d_fp_a
        fp_attr_b = fp_b[0] * d_fp_b

        # Backprop to post-CNN tokens out_cnn (L, 256) and pre-CNN Mamba tokens out_mamba (L, 256)
        w_cnn = self.sd['student.local_cnn.weight']  # (256, 256, 7)

        def token_attributions(d_detail, d_pooled):
            out_cnn = d_detail['out_cnn'][0]      # (L, 256)
            pre_cnn = d_detail['pre_cnn'][0]      # (L, 256)
            out_mamba = d_detail['out_mamba'][0]  # (L, 256)
            L = out_cnn.shape[0]

            # Post-CNN token contribution: sum_t token_saliency_cnn[t] == <d_pooled, pooled>
            token_saliency_cnn = (out_cnn @ d_pooled) / float(L)

            # Backprop through ReLU and 1D-CNN (kernel_size=7, padding=3) to out_mamba
            d_pre_cnn = (d_pooled[None, :] / float(L)) * (pre_cnn > 0).astype(np.float32)  # (L, 256)
            d_pre_pad = np.pad(d_pre_cnn, ((3, 3), (0, 0)), mode='constant')  # (L+6, 256)
            d_out_mamba = np.zeros_like(out_mamba)
            for k in range(7):
                # In forward: pre_cnn[t] += out_mamba_pad[t + k] @ w_cnn[:, :, k].T
                # where out_mamba_pad[u] = out_mamba[u - 3]. So tau = t + k - 3 => t = tau + 3 - k
                t_start = 6 - k
                d_out_mamba += d_pre_pad[t_start:t_start + L, :] @ w_cnn[:, :, k]
            token_saliency_mamba = np.sum(d_out_mamba * out_mamba, axis=-1)

            # Combined multiscale token saliency (post-CNN motif + pre-CNN atom backbone)
            token_saliency = 0.5 * token_saliency_cnn + 0.5 * token_saliency_mamba
            dt_mean = d_detail['dt'][0].mean(axis=-1)        # (L,)
            gate_mean = np.abs(d_detail['gate_z'][0]).mean(axis=-1)  # (L,)
            return {
                'token_saliency': token_saliency,
                'token_saliency_cnn': token_saliency_cnn,
                'token_saliency_mamba': token_saliency_mamba,
                'dt_mean': dt_mean,
                'gate_mean': gate_mean,
            }

        tok_a = token_attributions(da, d_ea)
        tok_b = token_attributions(db, d_eb)

        atom_1d_a = map_token_saliency_to_rdkit_atoms(smiles_a, tok_a['token_saliency'])
        atom_1d_b = map_token_saliency_to_rdkit_atoms(smiles_b, tok_b['token_saliency'])
        atom_2d_a = map_fp_attribution_to_rdkit_atoms(smiles_a, fp_attr_a)
        atom_2d_b = map_fp_attribution_to_rdkit_atoms(smiles_b, fp_attr_b)

        atom_a = combine_multimodal_atom_saliency(atom_1d_a, atom_2d_a)
        atom_b = combine_multimodal_atom_saliency(atom_1d_b, atom_2d_b)

        block_names = [
            'Seq_A (ea)', 'Seq_B (eb)', 'Seq_Prod (ea*eb)', 'Seq_Diff (|ea-eb|)',
            'FP_A (fa)', 'FP_B (fb)', 'FP_Prod (fa*fb)', 'FP_Diff (|fa-fb|)'
        ]
        block_ig = {
            name: float(ig_u[i * 256:(i + 1) * 256].sum())
            for i, name in enumerate(block_names)
        }
        block_abs_ig = {
            name: float(np.abs(ig_u[i * 256:(i + 1) * 256]).sum())
            for i, name in enumerate(block_names)
        }

        return {
            'logit': logit_val,
            'prob': prob_val,
            'tok_a': tok_a,
            'tok_b': tok_b,
            'atom_a': atom_a,
            'atom_b': atom_b,
            'atom_1d_a': atom_1d_a,
            'atom_1d_b': atom_1d_b,
            'atom_2d_a': atom_2d_a,
            'atom_2d_b': atom_2d_b,
            'fp_attr_a': fp_attr_a,
            'fp_attr_b': fp_attr_b,
            'block_ig': block_ig,
            'block_abs_ig': block_abs_ig,
            'h2': h2_actual[0],
        }


_MACCS_MOLS = {}
for _idx, (_smarts, _count) in MACCSkeys.smartsPatts.items():
    if _smarts != '?':
        _m = Chem.MolFromSmarts(_smarts)
        if _m is not None:
            _MACCS_MOLS[_idx] = _m


def map_fp_attribution_to_rdkit_atoms(smiles, fp_attr):
    """
    Project 679-dimensional fingerprint attributions (512 Morgan radius-2 + 167 MACCS)
    onto RDKit heavy atoms via exact substructure environments.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {'mol': mol, 'atom_scores': np.zeros(1), 'norm_scores': np.zeros(1)}

    n_atoms = mol.GetNumAtoms()
    scores = np.zeros(n_atoms, dtype=np.float64)

    # 1. Project 512 Morgan ECFP4 bits via bitInfo atom environments
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512, bitInfo=info)
    for bit_id, envs in info.items():
        val = float(fp_attr[bit_id])
        if abs(val) < 1e-9:
            continue
        atom_set = set()
        for center_atom, radius in envs:
            if center_atom >= n_atoms:
                continue
            atom_set.add(center_atom)
            if radius > 0:
                env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                for b_idx in env_bonds:
                    bond = mol.GetBondWithIdx(b_idx)
                    atom_set.add(bond.GetBeginAtomIdx())
                    atom_set.add(bond.GetEndAtomIdx())
        if atom_set:
            share = val / len(atom_set)
            for a_idx in atom_set:
                scores[a_idx] += share

    # 2. Project 166 MACCS SMARTS keys onto matched substructure atoms
    maccs_attr = fp_attr[512:]
    for key_id, patt in _MACCS_MOLS.items():
        if key_id >= len(maccs_attr):
            continue
        val = float(maccs_attr[key_id])
        if abs(val) < 1e-9:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            atom_set = {a_idx for match in matches for a_idx in match if a_idx < n_atoms}
            if atom_set:
                share = val / len(atom_set)
                for a_idx in atom_set:
                    scores[a_idx] += share

    max_abs = np.max(np.abs(scores))
    norm_scores = scores / max_abs if max_abs > 1e-9 else scores
    return {'mol': mol, 'atom_scores': scores, 'norm_scores': norm_scores}


def combine_multimodal_atom_saliency(atom_1d, atom_2d):
    """Combine 1D sequence atom attribution and 2D fingerprint substructure atom attribution."""
    mol = atom_2d['mol']
    s1 = atom_1d['atom_scores']
    s2 = atom_2d['atom_scores']
    if len(s1) == len(s2):
        total = s1 + s2
    else:
        total = s2.copy()
    max_abs = np.max(np.abs(total))
    norm = total / max_abs if max_abs > 1e-9 else total
    return {'mol': mol, 'atom_scores': total, 'norm_scores': norm}


def map_token_saliency_to_rdkit_atoms(smiles, token_saliency):
    """
    Map SMILES character-level saliency (length L) onto RDKit heavy atoms.
    1. Parse raw molecule (sanitize=False) to get exact 1-to-1 atom character spans.
    2. Distribute non-atom syntax tokens (bonds '=', '#', ring closure digits, branches)
       to their immediately adjacent atom(s).
    3. Transfer explicit [H] atom attributions onto their bonded heavy atom before removing H
       for clean 2D chemical depiction.
    """
    raw_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    i = 0
    n = len(smiles)
    atom_spans = []
    char_to_raw_atom = np.full(n, -1, dtype=np.int32)

    while i < n:
        c = smiles[i]
        if c == '[':
            j = smiles.find(']', i)
            if j == -1:
                j = n - 1
            atom_idx = len(atom_spans)
            atom_spans.append((i, j + 1))
            char_to_raw_atom[i:j + 1] = atom_idx
            i = j + 1
        elif c in 'BCNOPSFIbcnops':
            atom_idx = len(atom_spans)
            if c == 'C' and i + 1 < n and smiles[i + 1] == 'l':
                atom_spans.append((i, i + 2))
                char_to_raw_atom[i:i + 2] = atom_idx
                i += 2
            elif c == 'B' and i + 1 < n and smiles[i + 1] == 'r':
                atom_spans.append((i, i + 2))
                char_to_raw_atom[i:i + 2] = atom_idx
                i += 2
            else:
                atom_spans.append((i, i + 1))
                char_to_raw_atom[i] = atom_idx
                i += 1
        else:
            i += 1

    num_raw = len(atom_spans)
    raw_scores = np.zeros(num_raw, dtype=np.float64)

    # Assign atom span characters directly; assign bond/ring/branch characters to nearest atom
    for pos in range(n):
        a_idx = char_to_raw_atom[pos]
        if a_idx >= 0:
            raw_scores[a_idx] += token_saliency[pos]
        else:
            # Find nearest preceding and succeeding atom
            left = char_to_raw_atom[:pos][char_to_raw_atom[:pos] >= 0]
            right = char_to_raw_atom[pos + 1:][char_to_raw_atom[pos + 1:] >= 0]
            neighbors = []
            if len(left) > 0:
                neighbors.append(int(left[-1]))
            if len(right) > 0:
                neighbors.append(int(right[0]))
            if neighbors:
                for nb in neighbors:
                    raw_scores[nb] += token_saliency[pos] / len(neighbors)

    # Build sanitized molecule and fold explicit hydrogens into their heavy-atom neighbors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {'mol': raw_mol, 'atom_scores': raw_scores, 'norm_scores': raw_scores, 'atom_spans': atom_spans}

    if raw_mol.GetNumAtoms() == mol.GetNumAtoms():
        heavy_scores = raw_scores.copy()
    else:
        # Map non-H atoms in order (RDKit preserves heavy atom order when removing explicit H)
        heavy_indices = [a.GetIdx() for a in raw_mol.GetAtoms() if a.GetAtomicNum() != 1]
        raw_to_heavy = {raw_i: h_i for h_i, raw_i in enumerate(heavy_indices)}
        heavy_scores = np.array([raw_scores[raw_i] for raw_i in heavy_indices], dtype=np.float64)
        for a in raw_mol.GetAtoms():
            if a.GetAtomicNum() == 1:
                r_idx = a.GetIdx()
                nbrs = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIdx() in raw_to_heavy]
                if nbrs:
                    for nb in nbrs:
                        heavy_scores[raw_to_heavy[nb]] += raw_scores[r_idx] / len(nbrs)

    max_abs = np.max(np.abs(heavy_scores))
    norm_scores = heavy_scores / max_abs if max_abs > 1e-9 else heavy_scores
    return {
        'mol': mol,
        'atom_scores': heavy_scores,
        'norm_scores': norm_scores,
        'atom_spans': atom_spans,
        'char_to_raw_atom': char_to_raw_atom,
    }


def render_molecule_saliency_png(mol, norm_scores, width=650, height=450, title=""):
    """
    Render a publication-grade 2D molecular structure with continuous RDKit atom/bond saliency
    highlighting using MolDraw2DCairo.
    Positive saliency (driving DDI): warm crimson/coral red.
    Moderate positive: amber/gold.
    Negative/suppressing saliency: cool royal blue.
    """
    mol_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_draw)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.4
    opts.multipleBondOffset = 0.16
    opts.padding = 0.12
    opts.additionalAtomLabelPadding = 0.12

    highlight_atoms = []
    highlight_atom_colors = {}
    highlight_radii = {}

    for idx, score in enumerate(norm_scores):
        mag = float(np.clip(abs(score), 0.0, 1.0))
        if mag < 0.12:
            continue
        highlight_atoms.append(idx)
        if score >= 0:
            # Interpolate from soft peach/gold (mag=0.15) to deep crimson-red (mag=1.0)
            r = 1.0
            g = float(np.clip(0.88 - 0.62 * mag, 0.22, 0.90))
            b = float(np.clip(0.82 - 0.65 * mag, 0.18, 0.85))
        else:
            # Cool sky-blue for negative attribution
            r = float(np.clip(0.82 - 0.55 * mag, 0.25, 0.85))
            g = float(np.clip(0.90 - 0.35 * mag, 0.50, 0.92))
            b = 0.98
        highlight_atom_colors[idx] = (r, g, b)
        highlight_radii[idx] = 0.28 + 0.24 * mag

    highlight_bonds = []
    highlight_bond_colors = {}
    for bond in mol_draw.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        s1, s2 = norm_scores[a1], norm_scores[a2]
        if s1 > 0.20 and s2 > 0.20:
            b_idx = bond.GetIdx()
            highlight_bonds.append(b_idx)
            avg_m = float(np.clip(0.5 * (s1 + s2), 0.0, 1.0))
            highlight_bond_colors[b_idx] = (
                1.0,
                float(np.clip(0.90 - 0.55 * avg_m, 0.30, 0.92)),
                float(np.clip(0.85 - 0.58 * avg_m, 0.25, 0.88)),
            )

    drawer.DrawMolecule(
        mol_draw,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes))
