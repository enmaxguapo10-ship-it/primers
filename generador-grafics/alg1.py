"""
alg1.py – Eina d'investigació per a la distribució i patrons de primers
═══════════════════════════════════════════════════════════════════════════════════
Ús:
  python alg1.py            → menú interactiu
  alg1.py 500000     → analitza fins a 500,000
  alg1.py 1000 50000 → analitza de 1,000 a 50,000
"""

import sys
import time
import math
import collections
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓ ESTÈTICA
# ═══════════════════════════════════════════════════════════════════════════════

BG      = "#08080f"
PANEL   = "#0f0f1a"
ACCENT1 = "#00ffe0"
ACCENT2 = "#ff4d6d"
ACCENT3 = "#f5c542"
ACCENT4 = "#7b61ff"
ACCENT5 = "#39ff14"  
TEXT    = "#c8c8e0"
SUBTEXT = "#606080"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   "#2a2a40",
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       "#1e1e30",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

# ═══════════════════════════════════════════════════════════════════════════════
#  MILLER-RABIN OPTIMITZAT
# ═══════════════════════════════════════════════════════════════════════════════

_sieve = bytearray([1]) * 1500
_sieve[0] = _sieve[1] = 0
for _i in range(2, 1500):
    if _sieve[_i]:
        for _j in range(_i * _i, 1500, _i):
            _sieve[_j] = 0
_SMALL_PRIMES = [i for i in range(1500) if _sieve[i]]
_WITNESSES    = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _mr(n, a):
    if n % a == 0: return n == a
    d, r = n - 1, 0
    while not d & 1: d >>= 1; r += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1: return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1: return True
    return False

def es_primo(n):
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if p * p > n: return True
        if n % p == 0: return n == p
    return all(_mr(n, a) for a in _WITNESSES)

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITATS DE CONSOLA
# ═══════════════════════════════════════════════════════════════════════════════

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def color(text, codi):
    """Retorna text amb color ANSI."""
    codis = {
        'cyan':    '\033[96m',
        'magenta': '\033[95m',
        'groc':    '\033[93m',
        'verd':    '\033[92m',
        'vermell': '\033[91m',
        'blanc':   '\033[97m',
        'gris':    '\033[90m',
        'reset':   '\033[0m',
        'bold':    '\033[1m',
    }
    return f"{codis.get(codi,'')}{text}{codis['reset']}"

def barra_progres(actual, total, amplada=40, prefix='', sufix=''):
    """Mostra una barra de progrés animada amb detalls."""
    pct = actual / total if total > 0 else 0
    omplert = int(amplada * pct)
    barra = '█' * omplert + '░' * (amplada - omplert)
    pct_text = f"{pct*100:5.1f}%"
    print(f"\r  {color(prefix,'cyan')} [{color(barra,'magenta')}] {color(pct_text,'groc')} {sufix}",
          end='', flush=True)

def spinner_frame(i):
    frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
    return frames[i % len(frames)]

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓ DE PRIMERS AMB PROGRÉS
# ═══════════════════════════════════════════════════════════════════════════════

def generar_primers(lo, hi):
    """Genera primers amb barra de progrés i estadístiques en temps real."""
    rang = hi - max(2, lo) + 1
    print(f"\n  {color('Calculant primers','cyan')} en [{color(f'{lo:,}','groc')} – {color(f'{hi:,}','groc')}]...\n")
    
    t0 = time.perf_counter()
    prims = []
    bloc = max(1, rang // 100)  # actualitza cada 1%
    spin_i = 0
    
    for idx, n in enumerate(range(max(2, lo), hi + 1)):
        if es_primo(n):
            prims.append(n)
        
        if idx % bloc == 0 or idx == rang - 1:
            elapsed = time.perf_counter() - t0
            trobats = len(prims)
            vel = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (rang - idx - 1) / vel if vel > 0 else 0
            spin = spinner_frame(spin_i)
            spin_i += 1
            sufix = (f"{color(spin,'verd')} "
                     f"{color(f'{trobats:,} primers','blanc')} | "
                     f"{color(f'{vel:,.0f} n/s','gris')} | "
                     f"ETA {color(f'{eta:.1f}s','groc')}")
            barra_progres(idx + 1, rang, prefix='Calculant', sufix=sufix)
    
    elapsed = time.perf_counter() - t0
    print(f"\n\n  {color('✔','verd')} {color(f'{len(prims):,} primers trobats','blanc')} "
          f"en {color(f'{elapsed:.3f}s','groc')}\n")
    return prims

# ═══════════════════════════════════════════════════════════════════════════════
#  ANÀLISI DE DADES
# ═══════════════════════════════════════════════════════════════════════════════

def analitzar(prims):
    gaps = [prims[i+1] - prims[i] for i in range(len(prims)-1)]
    freq_gaps = collections.Counter(gaps)
    return gaps, freq_gaps

# ═══════════════════════════════════════════════════════════════════════════════
#  GRÀFICS
# ═══════════════════════════════════════════════════════════════════════════════

GRAFICS_DISPONIBLES = {
    1: "Densitat de primers (finestra lliscant)",
    2: "Distribució de gaps entre primers",
    3: "Gaps acumulats vs posició",
    4: "Mapa de calor de gaps",
    5: "Primers vs ln(n) — Teorema del nombre primer",
    6: "★ NOU: Promig de gap vs nombre primer (amb funció ln)",
}

def triar_grafics():
    """Permet a l'usuari triar quins gràfics vol generar."""
    print(f"\n  {color('═'*55,'gris')}")
    print(f"  {color('SELECCIÓ DE GRÀFICS','cyan')}")
    print(f"  {color('═'*55,'gris')}\n")
    
    for k, v in GRAFICS_DISPONIBLES.items():
        print(f"    {color(str(k),'groc')}. {v}")
    
    print(f"\n  {color('Opcions:','gris')}")
    print(f"    {color('tots','cyan')}     → genera tots els gràfics")
    print(f"    {color('1 3 5','cyan')}    → genera els gràfics 1, 3 i 5")
    print(f"    {color('cap','cyan')}      → surt sense generar gràfics\n")
    
    opcio = input(f"  {color('▶','magenta')} La teva selecció: ").strip().lower()
    
    if opcio == 'tots':
        return list(GRAFICS_DISPONIBLES.keys())
    elif opcio == 'cap':
        return []
    else:
        try:
            seleccio = [int(x) for x in opcio.split() if x.isdigit()]
            valids = [x for x in seleccio if x in GRAFICS_DISPONIBLES]
            if not valids:
                print(f"  {color('⚠ Cap selecció vàlida. Generant tots.','groc')}")
                return list(GRAFICS_DISPONIBLES.keys())
            return valids
        except:
            return list(GRAFICS_DISPONIBLES.keys())

def _guardar(fig, nom, idx, total):
    """Guarda un gràfic mostrant progrés."""
    path = Path(nom)
    print(f"  {color(spinner_frame(idx),'verd')} Guardant {color(nom,'cyan')}...", end='', flush=True)
    t0 = time.perf_counter()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    elapsed = time.perf_counter() - t0
    print(f"\r  {color('✔','verd')} Guardat  {color(nom,'cyan')} "
          f"{color(f'({elapsed:.2f}s)','gris')} "
          f"[{color(str(idx),'groc')}/{color(str(total),'groc')}]")
    return str(path)

# ── Gràfic 1: Densitat ────────────────────────────────────────────────────────
def grafic_densitat(prims, lo, hi):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Densitat de primers (finestra lliscant)", color=ACCENT1, fontsize=13)
    
    arr = np.array(prims)
    finestra = max(1, len(arr) // 80)
    x, y = [], []
    for i in range(0, len(arr) - finestra, finestra // 2):
        bloc = arr[i:i+finestra]
        x.append(bloc.mean())
        y.append(finestra / (bloc[-1] - bloc[0] + 1))
    
    ax.fill_between(x, y, alpha=0.3, color=ACCENT1)
    ax.plot(x, y, color=ACCENT1, linewidth=1.2)
    
    # Referència teòrica 1/ln(x)
    xr = np.linspace(max(2, lo), hi, 400)
    ax.plot(xr, 1/np.log(xr), '--', color=ACCENT3, linewidth=1, label='1/ln(x) teòric')
    ax.legend(fontsize=9)
    ax.set_xlabel("n"); ax.set_ylabel("densitat")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── Gràfic 2: Distribució de gaps ─────────────────────────────────────────────
def grafic_gaps_dist(gaps, freq_gaps):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Distribució de gaps entre primers consecutius", color=ACCENT2, fontsize=13)
    
    keys = sorted(freq_gaps.keys())
    vals = [freq_gaps[k] for k in keys]
    bars = ax.bar(keys, vals, color=ACCENT2, alpha=0.8, edgecolor=BG, linewidth=0.4)
    
    # Destacar el gap més freqüent
    max_idx = vals.index(max(vals))
    bars[max_idx].set_color(ACCENT3)
    bars[max_idx].set_alpha(1.0)
    ax.annotate(f"Gap {keys[max_idx]}\n(més freqüent)",
                xy=(keys[max_idx], vals[max_idx]),
                xytext=(keys[max_idx] + max(keys)*0.05, vals[max_idx]*0.9),
                color=ACCENT3, fontsize=8,
                arrowprops=dict(arrowstyle='->', color=ACCENT3))
    
    ax.set_xlabel("Gap"); ax.set_ylabel("Freqüència")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

# ── Gràfic 3: Gaps acumulats ──────────────────────────────────────────────────
def grafic_gaps_acum(prims, gaps):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Gaps acumulats vs posició del primer", color=ACCENT4, fontsize=13)
    
    x = np.array(prims[1:])
    y = np.cumsum(gaps)
    ax.plot(x, y, color=ACCENT4, linewidth=0.8, alpha=0.9)
    ax.fill_between(x, y, alpha=0.15, color=ACCENT4)
    ax.set_xlabel("Primer p"); ax.set_ylabel("Suma acumulada de gaps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── Gràfic 4: Mapa de calor ───────────────────────────────────────────────────
def grafic_heatmap(gaps, freq_gaps):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Mapa de calor de gaps", color=ACCENT3, fontsize=13)
    
    max_gap = min(max(freq_gaps.keys()), 200)
    mat = np.zeros((1, max_gap + 1))
    for g, f in freq_gaps.items():
        if g <= max_gap:
            mat[0, g] = f
    
    im = ax.imshow(mat, aspect='auto', cmap='plasma', norm=LogNorm(vmin=1, vmax=mat.max()))
    fig.colorbar(im, ax=ax, label='Freqüència (log)')
    ax.set_yticks([])
    ax.set_xlabel("Gap")
    fig.tight_layout()
    return fig

# ── Gràfic 5: Primers vs ln(n) ────────────────────────────────────────────────
def grafic_pnt(prims):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("π(x) vs x/ln(x)  —  Teorema del Nombre Primer", color=ACCENT1, fontsize=13)
    
    x = np.array(prims)
    pi_x = np.arange(1, len(prims) + 1)
    approx = x / np.log(x)
    
    ax.plot(x, pi_x, color=ACCENT1, linewidth=1.2, label='π(x) real')
    ax.plot(x, approx, '--', color=ACCENT3, linewidth=1, label='x/ln(x)')
    ax.legend(fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("π(x)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── Gràfic 6: Promig de gap vs nombre primer ────────────────────────────
def grafic_gap_promig_vs_prim(prims, gaps):
    """
    Mostra:
      • Punts de dispersió: cada primer p_i vs el seu gap g_i = p_{i+1} - p_i
      • Línia de promig suavitzat: promig mòbil de gaps
      • Línia teòrica: ln(p)  (predicció del Teorema del Nombre Primer)
    
    Funció representada: gap_promig(p) ≈ ln(p)
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Gap entre primers consecutius vs valor del primer",
                 color=ACCENT5, fontsize=14, fontweight='bold')
    
    p_vals = np.array(prims[:-1])   # p_i
    g_vals = np.array(gaps)          # gap_i = p_{i+1} - p_i
    
    # ── Punts: primers individuals ─────────────────────────────────────────
    # Submostrar si hi ha molts punts (per no saturar el gràfic)
    max_punts = 8000
    if len(p_vals) > max_punts:
        idx = np.random.choice(len(p_vals), max_punts, replace=False)
        idx.sort()
        px_plot, gx_plot = p_vals[idx], g_vals[idx]
    else:
        px_plot, gx_plot = p_vals, g_vals
    
    ax.scatter(px_plot, gx_plot,
               s=4, alpha=0.25, color=ACCENT4,
               label='Gap real (prim → prim+1)', zorder=2)
    
    # ── Promig mòbil ───────────────────────────────────────────────────────
    finestra = max(5, len(p_vals) // 120)
    kernel   = np.ones(finestra) / finestra
    g_suau   = np.convolve(g_vals, kernel, mode='valid')
    p_suau   = p_vals[finestra//2 : finestra//2 + len(g_suau)]
    
    ax.plot(p_suau, g_suau,
            color=ACCENT2, linewidth=2.0, alpha=0.9,
            label=f'Promig mòbil (finestra={finestra})', zorder=4)
    
    # ── Corba teòrica: ln(p) ───────────────────────────────────────────────
    p_teoria = np.linspace(p_vals[0] if p_vals[0] > 1 else 2, p_vals[-1], 600)
    g_teoria = np.log(p_teoria)
    
    ax.plot(p_teoria, g_teoria,
            '--', color=ACCENT3, linewidth=2.2, alpha=0.95,
            label='Funció teòrica:  gap_promig(p) ≈ ln(p)', zorder=5)
    
    # ── Anotació de la funció ──────────────────────────────────────────────
    mid_idx  = len(p_teoria) // 2
    mid_p    = p_teoria[mid_idx]
    mid_g    = g_teoria[mid_idx]
    ax.annotate("gap_promig(p) ≈ ln(p)",
                xy=(mid_p, mid_g),
                xytext=(mid_p * 0.6, mid_g * 1.6),
                color=ACCENT3, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ACCENT3, lw=1.5))
    
    # ── Llegenda i etiquetes ───────────────────────────────────────────────
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PANEL)
    ax.set_xlabel("Valor del primer  p", fontsize=11)
    ax.set_ylabel("Gap  g = p_next − p", fontsize=11)
    ax.grid(True, alpha=0.25)
    
    # Nota explicativa a peu de gràfic
    nota = ("El Teorema del Nombre Primer implica que la distància mitjana entre primers "
            "al voltant de p és aproximadament ln(p).\n"
            "Cada punt representa un primer real; la línia groga és la predicció teòrica.")
    fig.text(0.5, -0.03, nota, ha='center', fontsize=8, color=SUBTEXT, wrap=True)
    
    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓ AMB PROGRÉS
# ═══════════════════════════════════════════════════════════════════════════════

def generar_grafics(prims, gaps, freq_gaps, lo, hi, seleccio):
    """Genera i guarda els gràfics seleccionats mostrant progrés."""
    
    constructors = {
        1: ("densitat_primers.png",        lambda: grafic_densitat(prims, lo, hi)),
        2: ("gaps_distribucio.png",         lambda: grafic_gaps_dist(gaps, freq_gaps)),
        3: ("gaps_acumulats.png",           lambda: grafic_gaps_acum(prims, gaps)),
        4: ("gaps_heatmap.png",             lambda: grafic_heatmap(gaps, freq_gaps)),
        5: ("primers_vs_ln.png",            lambda: grafic_pnt(prims)),
        6: ("gap_promig_vs_primer.png",     lambda: grafic_gap_promig_vs_prim(prims, gaps)),
    }
    
    total = len(seleccio)
    arxius = []
    
    print(f"\n  {color('═'*55,'gris')}")
    print(f"  {color('GENERANT GRÀFICS','cyan')}  ({total} seleccionats)")
    print(f"  {color('═'*55,'gris')}\n")
    
    t_inici = time.perf_counter()
    
    for i, num in enumerate(seleccio, 1):
        nom, constructor = constructors[num]
        print(f"  {color(f'[{i}/{total}]','gris')} {color(GRAFICS_DISPONIBLES[num],'blanc')}")
        
        # Barra de "preparació"
        for pas in range(0, 101, 20):
            barra_progres(pas, 100, prefix='Preparant ', sufix='')
            time.sleep(0.03)
        print()
        
        fig = constructor()
        arxiu = _guardar(fig, nom, i, total)
        arxius.append(arxiu)
        print()
    
    t_total = time.perf_counter() - t_inici
    print(f"  {color('✔','verd')} Tots els gràfics generats en "
          f"{color(f'{t_total:.2f}s','groc')}\n")
    return arxius

# ═══════════════════════════════════════════════════════════════════════════════
#  PÀGINA D'AJUDA
# ═══════════════════════════════════════════════════════════════════════════════

def mostrar_ajuda():
    cls()
    print(f"""
  {color('═'*60,'cyan')}
  {color('  AJUDA — prime_research.py','blanc')}
  {color('═'*60,'cyan')}

  {color('DESCRIPCIÓ','groc')}
    Eina per analitzar la distribució dels nombres primers
    en un rang donat, generant gràfics estadístics.

  {color('ÚS DES DE LA LÍNIA DE COMANDES','groc')}
    {color('python prime_research.py','cyan')}
        → Obre el menú interactiu

    {color('python prime_research.py 500000','cyan')}
        → Analitza primers fins a 500,000

    {color('python prime_research.py 1000 50000','cyan')}
        → Analitza primers de 1,000 a 50,000

  {color('GRÀFICS DISPONIBLES','groc')}
    {color('1','magenta')}  Densitat de primers (finestra lliscant)
         Mostra quants primers hi ha per unitat en cada zona.
         Inclou la corba teòrica 1/ln(x).

    {color('2','magenta')}  Distribució de gaps
         Freqüència de cada distància entre primers consecutius.
         El gap més freqüent queda destacat en groc.

    {color('3','magenta')}  Gaps acumulats vs posició
         Suma acumulada dels gaps al llarg del rang.

    {color('4','magenta')}  Mapa de calor de gaps
         Visualització en color de la freqüència dels gaps.
         Escala logarítmica per apreciar valors rars.

    {color('5','magenta')}  Primers vs ln(n)  (Teorema del Nombre Primer)
         Compara π(x) real amb l'aproximació x/ln(x).

    {color('6','magenta')}  ★ Promig de gap vs nombre primer  (NOU)
         Cada punt és un primer real i el seu gap al següent.
         La línia groga mostra la funció teòrica:
         {color('gap_promig(p) ≈ ln(p)','groc')}
         derivada del Teorema del Nombre Primer.

  {color('CONCEPTES MATEMÀTICS','groc')}
    • {color('Gap','cyan')}: diferència entre dos primers consecutius.
      Exemple: gap(7,11) = 4
    • {color('Teorema del Nombre Primer','cyan')}: π(x) ≈ x/ln(x)
      implica que el gap promig al voltant de p és ≈ ln(p).
    • {color('Miller-Rabin','cyan')}: test de primalitat probabilístic
      (determinístic per als testimonis usats fins a 3.3·10²⁴).

  {color('ARXIUS GENERATS','groc')}
    Els gràfics es guarden com a .png al directori actual.

  {color('═'*60,'cyan')}
""")
    input(f"  {color('Prem Enter per tornar al menú...','gris')}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MENÚ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = f"""
  {color('╔══════════════════════════════════════════════════╗','cyan')}
  {color('║','cyan')}  {color('★  PRIME RESEARCH  ★','blanc')}                          {color('║','cyan')}
  {color('║','cyan')}  {color('Eina d\'anàlisi de nombres primers','gris')}              {color('║','cyan')}
  {color('╚══════════════════════════════════════════════════╝','cyan')}
"""

def menu_principal():
    while True:
        cls()
        print(BANNER)
        print(f"  {color('1','groc')}. Analitzar un rang de primers")
        print(f"  {color('2','groc')}. Ajuda")
        print(f"  {color('3','groc')}. Sortir\n")
        
        opcio = input(f"  {color('▶','magenta')} Opció: ").strip()
        
        if opcio == '1':
            executar_analisi()
        elif opcio == '2':
            mostrar_ajuda()
        elif opcio == '3':
            print(f"\n  {color('Fins aviat!','cyan')}\n")
            sys.exit(0)
        else:
            print(f"  {color('⚠ Opció no vàlida.','vermell')}")
            time.sleep(1)

def demanar_rang():
    """Demana el rang interactivament amb validació."""
    print(f"\n  {color('─'*50,'gris')}")
    print(f"  {color('CONFIGURACIÓ DEL RANG','cyan')}\n")
    
    while True:
        try:
            lo_str = input(f"  {color('Límit inferior','blanc')} (per defecte 2): ").strip()
            lo = int(lo_str) if lo_str else 2
            hi_str = input(f"  {color('Límit superior','blanc')} (ex: 100000): ").strip()
            hi = int(hi_str) if hi_str else 100000
            
            if lo < 2: lo = 2
            if hi <= lo:
                print(f"  {color('⚠ El límit superior ha de ser > límit inferior.','vermell')}")
                continue
            if hi - lo > 10_000_000:
                conf = input(f"  {color('⚠ Rang molt gran (>10M). Continuar? [s/N]: ','groc')}").strip().lower()
                if conf != 's':
                    continue
            return lo, hi
        except ValueError:
            print(f"  {color('⚠ Introdueix nombres enters vàlids.','vermell')}")

def executar_analisi():
    lo, hi = demanar_rang()
    
    # Calcular primers amb progrés
    prims = generar_primers(lo, hi)
    
    if len(prims) < 2:
        print(f"  {color('⚠ No hi ha prou primers en aquest rang per analitzar.','vermell')}")
        input(f"\n  {color('Prem Enter...','gris')}")
        return
    
    gaps, freq_gaps = analitzar(prims)
    
    # Estadístiques ràpides
    print(f"  {color('─'*50,'gris')}")
    print(f"  {color('Estadístiques:','cyan')}")
    print(f"    Primers trobats : {color(f'{len(prims):,}','blanc')}")
    print(f"    Gap mínim       : {color(str(min(gaps)),'verd')}")
    print(f"    Gap màxim       : {color(str(max(gaps)),'vermell')}")
    print(f"    Gap promig      : {color(f'{sum(gaps)/len(gaps):.2f}','groc')}")
    print(f"    ln(hi) teòric   : {color(f'{math.log(hi):.2f}','groc')}")
    print(f"  {color('─'*50,'gris')}\n")
    
    # Triar gràfics
    seleccio = triar_grafics()
    
    if not seleccio:
        print(f"\n  {color('Cap gràfic seleccionat. Tornant al menú.','gris')}")
        time.sleep(1.5)
        return
    
    # Generar gràfics
    arxius = generar_grafics(prims, gaps, freq_gaps, lo, hi, seleccio)
    
    print(f"  {color('Arxius generats:','cyan')}")
    for a in arxius:
        print(f"    {color('→','verd')} {a}")
    
    input(f"\n  {color('Prem Enter per tornar al menú...','gris')}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PUNT D'ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if len(args) == 0:
        # Mode interactiu amb menú
        menu_principal()
    
    elif len(args) == 1:
        hi = int(args[0])
        lo = 2
        prims = generar_primers(lo, hi)
        if len(prims) >= 2:
            gaps, freq_gaps = analitzar(prims)
            seleccio = triar_grafics()
            if seleccio:
                generar_grafics(prims, gaps, freq_gaps, lo, hi, seleccio)
    
    elif len(args) == 2:
        lo, hi = int(args[0]), int(args[1])
        prims = generar_primers(lo, hi)
        if len(prims) >= 2:
            gaps, freq_gaps = analitzar(prims)
            seleccio = triar_grafics()
            if seleccio:
                generar_grafics(prims, gaps, freq_gaps, lo, hi, seleccio)
