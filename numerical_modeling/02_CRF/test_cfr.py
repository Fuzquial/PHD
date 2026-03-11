"""
Reproduction de l'approche de Le Goc et al. (2015) - EUROCK
"Using correlated random fields for modeling the spatial heterogeneity of rock"

Ce script implémente :
1. Génération de champs aléatoires corrélés (CRF) log-normaux via FFT (méthode Gutjahr 1989)
2. Étude des propriétés de mise à l'échelle du CRF (μ et σ vs taille)
3. Simulation de tests UCS sur Masses Rocheuses Synthétiques (SRM) pseudo-numériques
4. Reproduction de l'effet d'échelle de la résistance UCS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter

# ─────────────────────────────────────────────────────────────
# 1. GÉNÉRATION DU CHAMP ALÉATOIRE CORRÉLÉ (CRF)
# ─────────────────────────────────────────────────────────────

def generate_gaussian_crf(nx, ny, nz, lambda_rf, domain_size):
    """
    Génère un champ gaussien corrélé via la méthode spectrale (Gutjahr 1989).
    Corrélation gaussienne : rho(h) = exp(-(h/lambda)^2)

    Paramètres
    ----------
    nx, ny, nz   : nombre de points de discrétisation
    lambda_rf    : longueur de corrélation [même unité que domain_size]
    domain_size  : taille physique du domaine (L x L x 2L) [m]

    Retour
    ------
    field : ndarray (nx, ny, nz) de valeurs gaussiennes centrées réduites
    """
    # Pas de discrétisation
    dx = domain_size[0] / nx
    dy = domain_size[1] / ny
    dz = domain_size[2] / nz

    # Fréquences spatiales
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    # Spectre de puissance de la corrélation gaussienne
    # S(k) = (lambda^3 * pi^(3/2)) * exp(-k^2 * lambda^2 / 4)
    S = (lambda_rf**3 * np.pi**1.5) * np.exp(-K2 * lambda_rf**2 / 4.0)

    # Champ gaussien blanc dans l'espace de Fourier
    noise = (np.random.randn(nx, ny, nz) + 1j * np.random.randn(nx, ny, nz)) / np.sqrt(2)

    # Filtrage par la racine du spectre
    field_f = np.sqrt(S) * noise

    # Retour dans l'espace réel
    field = np.real(np.fft.ifftn(field_f))

    # Normalisation : moyenne 0, écart-type 1
    field = (field - field.mean()) / (field.std() + 1e-12)

    return field


def gaussian_to_lognormal(field_gauss, mu_rf, sigma_rf):
    """
    Transformée en champ log-normal de moyenne mu_rf et écart-type sigma_rf.
    La transformation préserve la structure de corrélation spatiale.
    """
    # Paramètres de la gaussienne sous-jacente
    sigma_ln = np.sqrt(np.log(1 + (sigma_rf / mu_rf)**2))
    mu_ln    = np.log(mu_rf) - 0.5 * sigma_ln**2

    field_ln = np.exp(mu_ln + sigma_ln * field_gauss)
    return field_ln


# ─────────────────────────────────────────────────────────────
# 2. PROPRIÉTÉS DE MISE À L'ÉCHELLE DU CRF
# ─────────────────────────────────────────────────────────────

def compute_scaling_properties(field, domain_size, subsample_sizes, lambda_rf):
    """
    Calcule μ_i et σ_i dans des sous-échantillons cubiques de différentes tailles.

    Retour
    ------
    results : dict avec listes de (ls/lambda, mu_i/mu_L, sigma_i/sigma_L)
    """
    nx, ny, nz = field.shape
    Lx, Ly, Lz = domain_size

    mu_L    = field.mean()
    sigma_L = field.std()

    all_ls_norm   = []
    all_mu_norm   = []
    all_sigma_norm = []

    for ls in subsample_sizes:
        # Nombre de sous-domaines dans chaque direction
        n_sub_x = max(1, int(Lx / ls))
        n_sub_y = max(1, int(Ly / ls))
        n_sub_z = max(1, int(Lz / ls))

        # Taille en points
        pts_x = int(nx / n_sub_x)
        pts_y = int(ny / n_sub_y)
        pts_z = int(nz / n_sub_z)

        if pts_x < 2 or pts_y < 2 or pts_z < 2:
            continue

        for ix in range(n_sub_x):
            for iy in range(n_sub_y):
                for iz in range(n_sub_z):
                    sub = field[ix*pts_x:(ix+1)*pts_x,
                                iy*pts_y:(iy+1)*pts_y,
                                iz*pts_z:(iz+1)*pts_z]
                    mu_i    = sub.mean()
                    sigma_i = sub.std()
                    all_ls_norm.append(ls / lambda_rf)
                    all_mu_norm.append(mu_i / mu_L)
                    all_sigma_norm.append(sigma_i / sigma_L)

    return {
        'ls_norm'    : np.array(all_ls_norm),
        'mu_norm'    : np.array(all_mu_norm),
        'sigma_norm' : np.array(all_sigma_norm),
        'mu_L'       : mu_L,
        'sigma_L'    : sigma_L,
    }


def sigma_theoretical(l, lambda_rf, sigma_rf):
    """Équation (3) de l'article : σ(l) = sqrt(1 - exp(-(l/λ)²)) · σ_RF"""
    return np.sqrt(1 - np.exp(-(l / lambda_rf)**2)) * sigma_rf


# ─────────────────────────────────────────────────────────────
# 3. SIMULATION UCS (approche SRM simplifiée)
# ─────────────────────────────────────────────────────────────

def simulate_ucs(field, specimen_size, domain_size, alpha, beta, n_specimens=50):
    """
    Simule des tests UCS sur des sous-échantillons du champ CRF.
    UCS_i = alpha * (mu_i - sigma_i)^beta  [Équation (4)]

    Retour
    ------
    ucs_values : array des valeurs UCS simulées
    mu_vals, sigma_vals : statistiques locales correspondantes
    """
    nx, ny, nz = field.shape
    Lx, Ly, Lz = domain_size

    pts_x = max(2, int(nx * specimen_size / Lx))
    pts_y = max(2, int(ny * specimen_size / Ly))
    pts_z = max(2, int(nz * specimen_size / Lz))

    ucs_values = []
    mu_vals    = []
    sigma_vals = []

    max_ix = max(1, nx - pts_x)
    max_iy = max(1, ny - pts_y)
    max_iz = max(1, nz - pts_z)

    for _ in range(n_specimens):
        ix = np.random.randint(0, max_ix)
        iy = np.random.randint(0, max_iy)
        iz = np.random.randint(0, max_iz)

        sub = field[ix:ix+pts_x, iy:iy+pts_y, iz:iz+pts_z]
        mu_i    = sub.mean()
        sigma_i = sub.std()

        val = mu_i - sigma_i
        if val > 0:
            ucs = alpha * (val ** beta)
            ucs_values.append(ucs)
            mu_vals.append(mu_i)
            sigma_vals.append(sigma_i)

    return np.array(ucs_values), np.array(mu_vals), np.array(sigma_vals)


def ucs_theoretical(l, mu_L, sigma_L, lambda_rf, alpha, beta):
    """Équation (5) de l'article : UCS(l) = alpha * (mu_L - sigma_L * sqrt(1-exp(-(l/λ)²)))^beta"""
    sigma_eff = sigma_L * np.sqrt(1 - np.exp(-(l / lambda_rf)**2))
    val = mu_L - sigma_eff
    val = np.maximum(val, 1e-10)
    return alpha * (val ** beta)


# ─────────────────────────────────────────────────────────────
# 4. PARAMÈTRES ET EXÉCUTION PRINCIPALE
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # Paramètres des 5 CRF de l'article (Table 1)
    crf_params = {
        'T1': {'lambda_rf': 0.4, 'sigma_rf': 0.8, 'mu_rf': 1.0, 'color': 'royalblue',   'marker': 's'},
        'T2': {'lambda_rf': 0.2, 'sigma_rf': 0.4, 'mu_rf': 1.0, 'color': 'tomato',      'marker': '^'},
        'T3': {'lambda_rf': 0.8, 'sigma_rf': 0.4, 'mu_rf': 1.0, 'color': 'seagreen',    'marker': 'D'},
        'T4': {'lambda_rf': 0.4, 'sigma_rf': 0.2, 'mu_rf': 1.0, 'color': 'darkorange',  'marker': 'o'},
        'T5': {'lambda_rf': 0.4, 'sigma_rf': 0.8, 'mu_rf': 1.0, 'color': 'mediumpurple','marker': 'v'},
    }

    # Domaine : L=0.8 m, hauteur 2L
    L      = 0.8
    domain = (L, L, 2*L)
    nx, ny, nz = 80, 80, 160

    # Tailles de sous-échantillons (article : 0.1, 0.2, 0.4, 0.8 m)
    subsample_sizes = [0.1, 0.2, 0.4, 0.8]

    # Paramètres UCS empiriques (article : α=5.7e7, β=1.7)
    alpha = 5.7e7
    beta  = 1.7

    # ── Figure 1 : Visualisation du CRF (T1) ──
    print("Génération du CRF T1 pour visualisation...")
    p = crf_params['T1']
    field_gauss = generate_gaussian_crf(nx, ny, nz, p['lambda_rf'], domain)
    field_ln    = gaussian_to_lognormal(field_gauss, p['mu_rf'], p['sigma_rf'])

    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle("Figure 1 — Champ aléatoire corrélé T1 (λ=0.4 m, σ=0.8, μ=1.0)", fontsize=13)

    # Fonction de corrélation théorique
    h_vals = np.linspace(0, 1.0, 200)
    rho    = np.exp(-(h_vals / p['lambda_rf'])**2)
    axes[0].plot(h_vals, rho, 'royalblue', lw=2.5)
    axes[0].axvline(p['lambda_rf'], color='gray', ls='--', lw=1.2, label=f"λ = {p['lambda_rf']} m")
    axes[0].set_xlabel("Distance h [m]", fontsize=11)
    axes[0].set_ylabel("ρ(h)", fontsize=11)
    axes[0].set_title("Fonction de corrélation gaussienne")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Coupe 2D du champ
    im = axes[1].imshow(field_ln[:, :, nz//2].T, origin='lower',
                        extent=[0, L, 0, L], cmap='RdYlBu_r', aspect='equal')
    plt.colorbar(im, ax=axes[1], label='Valeur du champ (log-normal)')
    axes[1].set_title("Coupe centrale du champ CRF (plan XY)")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig('fig1_crf_visualisation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig1 sauvegardée")

    # ── Figure 2 : Propriétés de mise à l'échelle ──
    print("\nCalcul des propriétés de mise à l'échelle pour tous les CRF...")

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Figure 2 — Mise à l'échelle des statistiques du CRF (μ et σ)", fontsize=13)

    # Courbe théorique σ normalisée
    ls_norm_th = np.linspace(0.01, 4.0, 300)
    sigma_norm_th = np.sqrt(1 - np.exp(-ls_norm_th**2))
    axes2[1].plot(ls_norm_th, sigma_norm_th, 'k-', lw=2.5,
                  label=r'$f(l)=\sqrt{1-\exp(-(l/\lambda)^2)}$', zorder=5)

    for name, p in crf_params.items():
        print(f"  CRF {name}...")
        fg   = generate_gaussian_crf(nx, ny, nz, p['lambda_rf'], domain)
        fln  = gaussian_to_lognormal(fg, p['mu_rf'], p['sigma_rf'])
        res  = compute_scaling_properties(fln, domain, subsample_sizes, p['lambda_rf'])

        # Points individuels (transparents)
        axes2[0].scatter(res['ls_norm'], res['mu_norm'],
                         color=p['color'], alpha=0.2, s=15, marker=p['marker'])
        axes2[1].scatter(res['ls_norm'], res['sigma_norm'],
                         color=p['color'], alpha=0.2, s=15, marker=p['marker'])

        # Moyennes par taille
        for ls_val in np.unique(res['ls_norm']):
            mask = res['ls_norm'] == ls_val
            axes2[0].scatter(ls_val, res['mu_norm'][mask].mean(),
                             color=p['color'], s=80, marker=p['marker'],
                             edgecolors='k', linewidths=0.8,
                             label=name if ls_val == np.unique(res['ls_norm'])[0] else "")
            axes2[1].scatter(ls_val, res['sigma_norm'][mask].mean(),
                             color=p['color'], s=80, marker=p['marker'],
                             edgecolors='k', linewidths=0.8)

    for ax, ylabel, title in zip(
        axes2,
        [r'$\mu_i / \mu_L$', r'$\sigma_i / \sigma_L$'],
        ['Évolution de la moyenne normalisée', 'Évolution de l\'écart-type normalisé']
    ):
        ax.set_xlabel(r'$l_s / \lambda_{RF}$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title)
        ax.axhline(1.0, color='gray', ls='--', lw=1)
        ax.set_xlim(0, 4.2); ax.grid(True, alpha=0.3)

    axes2[0].set_ylim(0, 2.2)
    axes2[1].set_ylim(0, 1.6)
    axes2[0].legend(fontsize=9, ncol=2)
    axes2[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('fig2_scaling_properties.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig2 sauvegardée")

    # ── Figure 3a : Relation UCS = α(μ_i - σ_i)^β ──
    print("\nSimulation des tests UCS...")

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Figure 3 — Effet d'échelle UCS", fontsize=13)

    all_mu_sigma = []
    all_ucs      = []

    for name, p in crf_params.items():
        print(f"  CRF {name}...")
        fg  = generate_gaussian_crf(nx, ny, nz, p['lambda_rf'], domain)
        fln = gaussian_to_lognormal(fg, p['mu_rf'], p['sigma_rf'])

        for ls in subsample_sizes:
            ucs_vals, mu_vals, sigma_vals = simulate_ucs(
                fln, ls, domain, alpha, beta, n_specimens=40
            )
            x_vals = mu_vals - sigma_vals
            axes3[0].scatter(x_vals, ucs_vals, color=p['color'], alpha=0.5,
                             s=20, marker=p['marker'],
                             label=name if ls == subsample_sizes[0] else "")
            all_mu_sigma.extend(x_vals.tolist())
            all_ucs.extend(ucs_vals.tolist())

    # Droite théorique
    x_th = np.logspace(np.log10(0.1), np.log10(2.0), 200)
    y_th = alpha * x_th**beta
    axes3[0].plot(x_th, y_th, 'k--', lw=2.0, label=r'$y=\alpha(\mu_i-\sigma_i)^\beta$')
    axes3[0].set_xscale('log'); axes3[0].set_yscale('log')
    axes3[0].set_xlabel(r'$\mu_i - \sigma_i$', fontsize=12)
    axes3[0].set_ylabel('UCS [Pa]', fontsize=12)
    axes3[0].set_title(r'Relation UCS = α(μ$_i$ − σ$_i$)$^β$  — tous CRF, toutes tailles')
    axes3[0].legend(fontsize=9, ncol=2)
    axes3[0].grid(True, alpha=0.3, which='both')

    # ── Figure 3b : Effet d'échelle pour T1 ──
    p1 = crf_params['T1']
    fg1  = generate_gaussian_crf(nx, ny, nz, p1['lambda_rf'], domain)
    fln1 = gaussian_to_lognormal(fg1, p1['mu_rf'], p1['sigma_rf'])
    mu_L    = fln1.mean()
    sigma_L = fln1.std()

    sizes_plot = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
    ucs_avg = []
    ucs_all_x = []
    ucs_all_y = []

    for ls in sizes_plot:
        ucs_vals, _, _ = simulate_ucs(fln1, ls, domain, alpha, beta, n_specimens=60)
        if len(ucs_vals) > 0:
            ucs_avg.append((ls, ucs_vals.mean()))
            ucs_all_x.extend([ls]*len(ucs_vals))
            ucs_all_y.extend(ucs_vals.tolist())

    axes3[1].scatter(ucs_all_x, ucs_all_y, color='royalblue', alpha=0.25, s=15, label='sous-échantillons')
    if ucs_avg:
        xs, ys = zip(*ucs_avg)
        axes3[1].plot(xs, ys, 'bs-', ms=7, lw=2, label='moyenne')

    # Courbe théorique (Éq. 5)
    l_th  = np.linspace(0.02, 0.85, 200)
    ucs_th = ucs_theoretical(l_th, mu_L, sigma_L, p1['lambda_rf'], alpha, beta)
    axes3[1].plot(l_th, ucs_th, 'k--', lw=2.0, label='Éq. (5) théorique')

    axes3[1].axvline(p1['lambda_rf'], color='darkorange', ls=':', lw=2.0,
                     label=f"λ_RF = {p1['lambda_rf']} m")
    axes3[1].axhline(alpha * (mu_L - sigma_L)**beta, color='steelblue', ls=':', lw=1.5,
                     label=f"σ_RF plateau")

    axes3[1].set_xlabel('Taille de l\'éprouvette [m]', fontsize=12)
    axes3[1].set_ylabel('UCS [Pa]', fontsize=12)
    axes3[1].set_title('Effet d\'échelle UCS — CRF T1 (λ=0.4 m)')
    axes3[1].legend(fontsize=9)
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig3_ucs_scale_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig3 sauvegardée")

    # ── Figure bonus : Coupe 3D isométrique du champ CRF ──
    fig4 = plt.figure(figsize=(10, 7))
    ax4  = fig4.add_subplot(111, projection='3d')
    fig4.suptitle("Champ CRF T1 — Coupes orthogonales (vue 3D)", fontsize=13)

    step = 4
    X = np.linspace(0, L, nx//step)
    Y = np.linspace(0, L, ny//step)
    Z = np.linspace(0, 2*L, nz//step)

    fln_ds = fln1[::step, ::step, ::step]

    # Plan XY (z=0)
    Xg, Yg = np.meshgrid(X, Y, indexing='ij')
    ax4.contourf(Xg, Yg, fln_ds[:,:,0], zdir='z', offset=0,
                 cmap='RdYlBu_r', alpha=0.7, levels=15)
    # Plan XZ (y=max)
    Xg2, Zg2 = np.meshgrid(X, Z, indexing='ij')
    ax4.contourf(Xg2, Zg2, fln_ds[:,ny//step-1,:], zdir='y', offset=L,
                 cmap='RdYlBu_r', alpha=0.7, levels=15)
    # Plan YZ (x=0)
    Yg3, Zg3 = np.meshgrid(Y, Z, indexing='ij')
    ax4.contourf(Yg3, Zg3, fln_ds[0,:,:], zdir='x', offset=0,
                 cmap='RdYlBu_r', alpha=0.7, levels=15)

    ax4.set_xlabel('x [m]'); ax4.set_ylabel('y [m]'); ax4.set_zlabel('z [m]')
    ax4.set_xlim(0, L); ax4.set_ylim(0, L); ax4.set_zlim(0, 2*L)
    ax4.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.savefig('fig4_3d_crf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig4 sauvegardée")

    print("\n✓ Toutes les figures générées avec succès.")


if __name__ == "__main__":
    main()