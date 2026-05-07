"""
requirements.py — Instal·la automàticament les dependències de primes.py
Executa amb: python requirements.py
"""

import subprocess
import sys

PACKAGES = [
    "matplotlib",
]

def install(package):
    print(f"  → Instal·lant {package}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ {package} instal·lat correctament")
    else:
        print(f"  ✗ Error instal·lant {package}:\n{result.stderr}")
        sys.exit(1)

def check(package):
    """Retorna True si el paquet ja està instal·lat."""
    import importlib.util
    # matplotlib s'importa com 'matplotlib', no cal mapeig especial aquí
    return importlib.util.find_spec(package) is not None

if __name__ == "__main__":
    print("=== Comprovant dependències de primes.py ===\n")
    for pkg in PACKAGES:
        if check(pkg):
            print(f"  ✓ {pkg} ja està instal·lat")
        else:
            install(pkg)
    print("\n✓ Tot llest. Ja pots executar: python primes.py")